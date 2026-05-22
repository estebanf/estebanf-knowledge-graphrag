"""CLI config: ``~/.config/rag/config.toml`` with env-var overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tomli_w


DEFAULT_PATH = Path.home() / ".config" / "rag" / "config.toml"


class CliConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class CliConfig:
    server_url: Optional[str]
    api_key: Optional[str]


def _from_file(path: Path) -> CliConfig:
    if not path.is_file():
        return CliConfig(server_url=None, api_key=None)
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return CliConfig(server_url=None, api_key=None)
    server = data.get("server_url")
    api_key = data.get("api_key")
    return CliConfig(
        server_url=str(server) if server else None,
        api_key=str(api_key) if api_key else None,
    )


def load_cli_config(*, path: Path | None = None) -> CliConfig:
    """Resolve precedence: env > file > none."""
    path = path or DEFAULT_PATH
    file_cfg = _from_file(path)
    server = os.environ.get("RAG_SERVER_URL") or file_cfg.server_url
    api_key = os.environ.get("RAG_API_KEY") or file_cfg.api_key
    return CliConfig(server_url=server, api_key=api_key)


def save_cli_config(cfg: CliConfig, *, path: Path | None = None) -> None:
    path = path or DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, str] = {}
    if cfg.server_url:
        data["server_url"] = cfg.server_url
    if cfg.api_key:
        data["api_key"] = cfg.api_key
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def require_config(*, path: Path | None = None) -> CliConfig:
    cfg = load_cli_config(path=path or DEFAULT_PATH)
    if not cfg.server_url or not cfg.api_key:
        raise CliConfigError(
            "CLI is not configured. Run `rag configure` or set RAG_SERVER_URL and RAG_API_KEY."
        )
    return cfg
