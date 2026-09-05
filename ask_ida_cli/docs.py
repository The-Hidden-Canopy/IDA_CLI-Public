"""Bundled public documentation explanation surface."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any


def _load() -> list[dict[str, Any]]:
    raw = files("ask_ida_cli").joinpath("public_docs.json").read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, list):
        raise TypeError("public documentation bundle must be a list")
    return [item for item in value if isinstance(item, dict)]


def bundled_sha256() -> str:
    payload = files("ask_ida_cli").joinpath("public_docs.json").read_bytes()
    return hashlib.sha256(payload).hexdigest()


def verify_bundled_docs(catalog: dict[str, Any]) -> None:
    expected = str((catalog.get("docs_bundle") or {}).get("sha256") or "")
    if not expected or expected != bundled_sha256():
        raise ValueError("bundled public documentation does not match the catalog attestation")


def explain(query: str, *, limit: int = 3) -> list[dict[str, str]]:
    needle = str(query or "").strip().casefold()
    if not needle:
        return []
    terms = {term for term in needle.split() if len(term) > 2}
    ranked: list[tuple[int, dict[str, str]]] = []
    for item in _load():
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        haystack = f"{title} {body}".casefold()
        score = sum(haystack.count(term) for term in terms)
        if score:
            ranked.append((score, {"title": title, "body": body, "source": str(item.get("source") or "public-docs")}))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["title"].casefold()))
    return [item for _, item in ranked[: max(1, min(limit, 10))]]
