"""Explicit public CLI settings with no private training-root default."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


DEFAULT_CATALOG_URL = "https://www.the-hidden-canopy.org/api/ask-ida/cli/catalog"


def _default_catalog_public_key() -> str | None:
    try:
        return files("ask_ida_cli").joinpath("catalog_public_key.pem").read_text(encoding="ascii")
    except (FileNotFoundError, OSError):
        return None


def _resolve(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


@dataclass(frozen=True)
class Settings:
    root: Path
    model_root: Path
    catalog_url: str = DEFAULT_CATALOG_URL
    catalog_public_key: str | None = None
    timeout_seconds: float = 10.0


def load_settings(
    *,
    root: str | Path | None = None,
    model_root: str | Path | None = None,
    catalog_url: str | None = None,
    catalog_public_key: str | None = None,
    timeout_seconds: float | None = None,
) -> Settings:
    resolved_root = _resolve(root or os.environ.get("ASK_IDA_PUBLIC_ROOT") or Path.cwd())
    resolved_model_root = _resolve(
        model_root or os.environ.get("ASK_IDA_PUBLIC_MODEL_ROOT") or resolved_root / "models"
    )
    timeout = float(timeout_seconds or os.environ.get("ASK_IDA_PUBLIC_TIMEOUT", "10"))
    if timeout <= 0 or timeout > 60:
        raise ValueError("timeout must be between 0 and 60 seconds")
    return Settings(
        root=resolved_root,
        model_root=resolved_model_root,
        catalog_url=str(catalog_url or os.environ.get("ASK_IDA_PUBLIC_CATALOG_URL") or DEFAULT_CATALOG_URL),
        catalog_public_key=catalog_public_key or os.environ.get("ASK_IDA_PUBLIC_CATALOG_PUBLIC_KEY") or _default_catalog_public_key(),
        timeout_seconds=timeout,
    )
