"""Conversation checkpoints plus a cross-conversation semantic store.

Short-term memory is a SQLite checkpointer keyed by thread id.
Long-term memory is a SQLite store: preferences and analysis findings,
retrieved by meaning with a local embedding model. DeepSeek does not
provide an embeddings API, so vectors are computed on this machine.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from data_agent.config import settings

FINDINGS = ("findings",)
PREFERENCES = ("preferences",)
THREADS = ("threads",)
EMBEDDING_DIMS = 384
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EmbedFn = Callable[[Sequence[str]], list[list[float]]]
_embedder: Any = None


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed with FastEmbed. The model downloads once, on first use."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return [vector.tolist() for vector in _embedder.embed(list(texts))]


@dataclass
class Memory:
    checkpointer: SqliteSaver
    store: SqliteStore
    _contexts: tuple[Iterator[Any], Iterator[Any]]

    def close(self) -> None:
        for context in self._contexts:
            context.__exit__(None, None, None)

    def save_finding(self, title: str, finding: str, dataset: str = "") -> str:
        return put_finding(self.store, title, finding, dataset)

    def save_preference(self, preference: str) -> str:
        return put_preference(self.store, preference)

    def remember_thread(self, thread_id: str, title: str) -> None:
        self.store.put(
            THREADS,
            thread_id,
            {"text": title.strip() or "New conversation", "title": title.strip() or "New conversation", "updated_at": _now()},
            index=False,
        )

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        findings = self._search(FINDINGS, query, limit)
        preferences = self._search(PREFERENCES, query, min(3, limit))
        combined = findings + preferences
        combined.sort(key=lambda item: item["score"], reverse=True)
        return combined[:limit]

    def list_preferences(self) -> list[dict[str, Any]]:
        return self._search(PREFERENCES, query=None, limit=20)

    def list_threads(self) -> list[dict[str, str]]:
        items = self._search(THREADS, query=None, limit=30)
        items.sort(key=lambda item: item["value"].get("updated_at", ""), reverse=True)
        return [
            {"id": item["key"], "title": item["value"].get("title") or item["key"]}
            for item in items
        ]

    def prompt_block(self, query: str) -> str:
        if not query.strip():
            return ""
        lines = []
        for item in self.list_preferences()[:8]:
            lines.append(f"- preference: {item['value'].get('text', '')}")
        for item in self._search(FINDINGS, query, limit=4):
            title = item["value"].get("title") or "finding"
            text = item["value"].get("text", "")
            lines.append(f"- finding ({title}): {text}")
        if not lines:
            return ""
        return "## Memories from earlier conversations\n" + "\n".join(lines)

    def _search(self, namespace: tuple[str, ...], query: str | None, limit: int) -> list[dict[str, Any]]:
        items = self.store.search(namespace, query=query or None, limit=limit)
        results = []
        for item in items:
            results.append(
                {
                    "namespace": "/".join(item.namespace),
                    "key": item.key,
                    "score": float(item.score or 0),
                    "value": dict(item.value),
                }
            )
        return results


def open_memory(
    directory: Any = None,
    embed: EmbedFn | None = None,
    dims: int = EMBEDDING_DIMS,
) -> Memory:
    folder = settings.memory_dir if directory is None else directory
    folder.mkdir(parents=True, exist_ok=True)
    index = {
        "dims": dims,
        "embed": embed or embed_texts,
        "fields": ["text"],
    }
    saver_cm = SqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite"))
    store_cm = SqliteStore.from_conn_string(str(folder / "store.sqlite"), index=index)
    checkpointer = saver_cm.__enter__()
    store = store_cm.__enter__()
    checkpointer.setup()
    store.setup()
    return Memory(checkpointer, store, (saver_cm, store_cm))


@contextmanager
def memory_session(directory: Any, embed: EmbedFn, dims: int = 8) -> Iterator[Memory]:
    memory = open_memory(directory, embed=embed, dims=dims)
    try:
        yield memory
    finally:
        memory.close()


def put_finding(store: Any, title: str, finding: str, dataset: str = "") -> str:
    key = uuid4().hex
    store.put(
        FINDINGS,
        key,
        {
            "text": _memory_text(title, finding, dataset),
            "title": title.strip(),
            "dataset": dataset.strip(),
            "created_at": _now(),
        },
    )
    return key


def put_preference(store: Any, preference: str) -> str:
    text = preference.strip()
    key = sha256(text.encode()).hexdigest()[:16]
    store.put(
        PREFERENCES,
        key,
        {"text": text, "title": text[:80], "created_at": _now()},
    )
    return key


def _memory_text(title: str, finding: str, dataset: str) -> str:
    parts = [title.strip(), finding.strip()]
    if dataset.strip():
        parts.append(f"Dataset: {dataset.strip()}")
    return "\n".join(part for part in parts if part)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
