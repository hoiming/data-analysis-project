"""LangGraph analyst: tool-calling loop, thread checkpoints, and long-term memory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from data_agent.charts import is_artifact
from data_agent.config import settings
from data_agent.memory import Memory, open_memory
from data_agent.prompts import SYSTEM_PROMPT
from data_agent.tools import build_tools


@dataclass
class Turn:
    text: str
    charts: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


class Analyst:
    def __init__(self, memory: Memory | None = None) -> None:
        self.memory = memory or open_memory()
        model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key or "not-set",
            base_url=settings.base_url,
            temperature=0,
            timeout=90,
            max_retries=1,
            use_responses_api=False,
        )
        self.graph = create_agent(
            model=model,
            tools=build_tools(),
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.memory.checkpointer,
            store=self.memory.store,
            middleware=[_memory_prompt(self.memory)],
        )

    def ask(self, thread_id: str, message: str) -> Turn:
        if not settings.api_key:
            return Turn("Set DEEPSEEK_API_KEY in .env, then start the app again.")
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(settings.max_agent_steps * 2, 8),
        }
        try:
            result = self.graph.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config,
            )
        except Exception as exc:  # surface provider and tool failures in the chat
            return Turn(f"The agent stopped: {exc}")

        self.memory.remember_thread(thread_id, message.strip()[:80])
        return _turn_from_messages(result.get("messages", []))

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = (snapshot.values or {}).get("messages", []) if snapshot else []
        return _gradio_history(messages)

    def threads(self) -> list[dict[str, str]]:
        return self.memory.list_threads()


def _memory_prompt(memory: Memory):
    @dynamic_prompt
    def with_memories(request: ModelRequest) -> str:
        query = _latest_user_text(request.messages)
        try:
            block = memory.prompt_block(query)
        except Exception:
            block = ""
        if not block:
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\n\n{block}"

    return with_memories


def _turn_from_messages(messages: list[Any]) -> Turn:
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index + 1
    turn_messages = messages[start:]
    tools = [message.name for message in turn_messages if isinstance(message, ToolMessage) and message.name]
    charts = []
    for message in turn_messages:
        if isinstance(message, ToolMessage) and message.name == "plot_dataset":
            charts.extend(_chart_paths(message.content))
    text = ""
    for message in reversed(turn_messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = _text(message).strip()
            if text:
                break
    if not text:
        text = "I did not produce a final answer. Try a more specific question."
    if tools:
        text = f"Tools: {', '.join(tools)}\n\n{text}"
    return Turn(text=text, charts=charts, tools=tools)


def _gradio_history(messages: list[Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    charts: list[str] = []
    pending_user = False
    for message in messages:
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": _text(message)})
            charts = []
            pending_user = True
        elif isinstance(message, ToolMessage) and message.name == "plot_dataset":
            charts.extend(_chart_paths(message.content))
        elif isinstance(message, AIMessage) and not message.tool_calls and pending_user:
            content: list[Any] = []
            text = _text(message).strip()
            if text:
                content.append(text)
            content.extend(_file_content(path) for path in charts)
            if content:
                history.append({"role": "assistant", "content": content if len(content) > 1 else content[0]})
            charts = []
            pending_user = False
    return history


def _chart_paths(content: Any) -> list[str]:
    text = content if isinstance(content, str) else _text_from_content(content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    path = payload.get("path") if isinstance(payload, dict) else None
    if isinstance(path, str) and is_artifact(path):
        return [path]
    return []


def _file_content(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "orig_name": Path(path).name,
        "mime_type": "image/png",
        "meta": {"_type": "gradio.FileData"},
    }


def _latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            return _text(message)
    return ""


def _text(message: Any) -> str:
    return _text_from_content(getattr(message, "content", ""))


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(content)
