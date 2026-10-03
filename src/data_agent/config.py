"""Runtime paths and DeepSeek connection settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

_PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


def normalize_proxy_env() -> None:
    """Rewrite socks:// to socks5:// so httpx can use a local SOCKS proxy.

    Some clients export ALL_PROXY=socks://127.0.0.1:10808/. httpx only accepts
    http, https, socks5, and socks5h.
    """
    for key in _PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        stripped = value.strip()
        if stripped.lower().startswith("socks://"):
            os.environ[key] = "socks5://" + stripped[len("socks://") :]


normalize_proxy_env()


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else default
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    data_dir: Path
    artifacts_dir: Path
    memory_dir: Path
    max_preview_rows: int = 30
    max_agent_steps: int = 12

    @classmethod
    def load(cls) -> Settings:
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            data_dir=_path_from_env("DATA_DIR", PROJECT_ROOT / "data"),
            artifacts_dir=_path_from_env("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"),
            memory_dir=_path_from_env("MEMORY_DIR", PROJECT_ROOT / "memory"),
        )


settings = Settings.load()
