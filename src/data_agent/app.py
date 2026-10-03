"""Gradio chat for the data analysis agent."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import gradio as gr
import pandas as pd

from data_agent.agent import Analyst
from data_agent.config import settings
from data_agent.datasets import SUPPORTED_SUFFIXES, list_dataset_files

_analyst: Analyst | None = None


def get_analyst() -> Analyst:
    global _analyst
    if _analyst is None:
        _analyst = Analyst()
    return _analyst


def dataset_table() -> pd.DataFrame:
    files = list_dataset_files()
    if not files:
        return pd.DataFrame(columns=["name", "rows", "columns", "bytes"])
    return pd.DataFrame(files)


def thread_update():
    return gr.update(choices=thread_choices())


def thread_choices() -> list[tuple[str, str]]:
    try:
        threads = get_analyst().threads()
    except Exception:
        threads = []
    choices = [("New conversation", "")]
    choices.extend((f"{item['title']}", item["id"]) for item in threads)
    return choices


def save_uploads(files: list[str] | str | None) -> pd.DataFrame:
    if not files:
        return dataset_table()
    paths = files if isinstance(files, list) else [files]
    for raw in paths:
        source = Path(raw)
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        target = settings.data_dir / source.name
        if target.exists() and target.resolve() != source.resolve():
            target = settings.data_dir / f"{source.stem}-{uuid4().hex[:6]}{source.suffix}"
        if source.resolve() != target.resolve():
            shutil.copy(source, target)
    return dataset_table()


def respond(message: str, history: list[dict], thread_id: str):
    history = list(history or [])
    text = (message or "").strip()
    if not text:
        yield history, "", thread_id, gr.skip()
        return
    if not thread_id:
        thread_id = uuid4().hex
    history.append({"role": "user", "content": text})
    yield history, "", thread_id, gr.skip()
    try:
        turn = get_analyst().ask(thread_id, text)
    except Exception as exc:
        history.append({"role": "assistant", "content": f"The agent stopped: {exc}"})
        yield history, "", thread_id, gr.skip()
        return
    content: list[object] = [turn.text]
    for path in turn.charts:
        content.append(
            {
                "path": path,
                "orig_name": Path(path).name,
                "mime_type": "image/png",
                "meta": {"_type": "gradio.FileData"},
            }
        )
    history.append({"role": "assistant", "content": content if len(content) > 1 else content[0]})
    yield history, "", thread_id, thread_update()


def load_thread(thread_id: str):
    if not thread_id:
        return gr.skip(), gr.skip()
    return get_analyst().history(thread_id), thread_id


def new_thread():
    return [], uuid4().hex, gr.update(value=None)


def build() -> gr.Blocks:
    key_note = (
        f"Model `{settings.model}` via `{settings.base_url}`."
        if settings.api_key
        else "Add DEEPSEEK_API_KEY to .env before chatting. Dataset tools still work from the file list."
    )
    with gr.Blocks(title="Data analysis agent") as demo:
        gr.Markdown(
            f"""# Data analysis agent
Ask questions about CSV, Excel, Parquet, or JSON files in `data/`.
The agent calls tools, remembers this conversation, and can recall findings from earlier ones.

{key_note}"""
        )
        thread_id = gr.State("")
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="Drop files into the data folder",
                    file_count="multiple",
                    file_types=[".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"],
                    type="filepath",
                    height=140,
                )
                datasets = gr.Dataframe(value=dataset_table, label="Datasets", interactive=False)
                threads = gr.Dropdown(label="Conversations", choices=thread_choices(), value=None)
                refresh_threads = gr.Button("Refresh conversations")
                new_chat = gr.Button("New conversation")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Analyst", height=420)
                message = gr.Textbox(
                    label="Question",
                    placeholder="Which region has the highest revenue in sample_sales.csv?",
                    lines=2,
                )
                send = gr.Button("Send", variant="primary")

        upload.change(save_uploads, inputs=upload, outputs=datasets)
        send.click(
            respond,
            inputs=[message, chatbot, thread_id],
            outputs=[chatbot, message, thread_id, threads],
        )
        message.submit(
            respond,
            inputs=[message, chatbot, thread_id],
            outputs=[chatbot, message, thread_id, threads],
        )
        refresh_threads.click(thread_update, outputs=threads)
        threads.change(load_thread, inputs=threads, outputs=[chatbot, thread_id])
        new_chat.click(new_thread, outputs=[chatbot, thread_id, threads])
    return demo


def main() -> None:
    build().launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
