"""Read and validate the public, confirmation-gated Neural Forge leaderboard."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from .errors import CLIError

SCHEMA_VERSION = "neural-forge-leaderboard.v1"
DEFAULT_LIMIT = 20
MAX_LIMIT = 100
MAX_OFFSET = 100_000
MAX_TOTAL = 10_000
MAX_CONFIGURATIONS = 10_000
MAX_ENTRIES_PER_CONFIGURATION = 5
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_TEXT = 128
MAX_SCORE = 1e12
MAX_THROUGHPUT = 1e9
PUBLIC_VIEWS = frozenset({"throughput", "outcome"})
PUBLIC_ENTRY_BASE_KEYS = frozenset({
    "rank", "entry_id", "user_id", "model_id", "revision", "gpu", "published_at", "view",
})
PUBLIC_VIEW_KEYS = frozenset({"entries", "total", "offset", "limit", "configurations"})
PUBLIC_GPU_KEYS = frozenset({"vendor", "model"})
PUBLIC_CONFIGURATION_KEYS = frozenset({"model_id", "revision", "gpu", "total", "entries"})
OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MODEL_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+/-]{0,127}$")
ENTRY_ID = re.compile(r"^nflb-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", re.IGNORECASE)
USER_ID = re.compile(r"^operator-[a-f0-9]{12}$")


class LeaderboardError(CLIError):
    """A public leaderboard response or transport is invalid."""

    def __init__(self, message: str, *, code: str = "leaderboard_invalid") -> None:
        super().__init__(message, code=code, exit_code=3)


class _HTTPSOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only when the HTTPS origin remains unchanged."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        try:
            source = urlsplit(req.full_url)
            target = urlsplit(newurl)
            source_host = source.hostname
            target_host = target.hostname
            source_port = source.port or 443
            target_port = target.port or 443
        except ValueError as exc:
            raise LeaderboardError("leaderboard redirect URL is invalid", code="leaderboard_transport") from exc
        if (
            source.scheme.casefold() != "https"
            or target.scheme.casefold() != "https"
            or not source_host
            or not target_host
            or source_host.casefold() != target_host.casefold()
            or source_port != target_port
            or target.username
            or target.password
            or target.fragment
        ):
            raise LeaderboardError(
                "leaderboard redirect must remain on the original HTTPS origin",
                code="leaderboard_transport",
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LeaderboardError(f"leaderboard {label} must be an object", code="leaderboard_schema")
    return value


def _exact_keys(value: Any, allowed: frozenset[str], label: str) -> dict[str, Any]:
    item = _object(value, label)
    if set(item) != allowed:
        raise LeaderboardError(f"leaderboard {label} contains an unexpected field", code="leaderboard_schema")
    return item


def _text(value: Any, label: str, pattern: re.Pattern[str] = MODEL_TEXT) -> str:
    if not isinstance(value, str) or len(value) < 1 or len(value) > MAX_TEXT or not pattern.fullmatch(value):
        raise LeaderboardError(f"leaderboard {label} is invalid", code="leaderboard_schema")
    return value


def _opaque(value: Any, label: str) -> str:
    result = _text(value, label, OPAQUE_ID)
    if ".." in result or result.casefold() in {"__proto__", "prototype", "constructor"}:
        raise LeaderboardError(f"leaderboard {label} is invalid", code="leaderboard_schema")
    return result


def _number(value: Any, label: str, *, maximum: float = MAX_SCORE) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LeaderboardError(f"leaderboard {label} is invalid", code="leaderboard_schema")
    if abs(float(value)) > maximum:
        raise LeaderboardError(f"leaderboard {label} is invalid", code="leaderboard_schema")
    return value


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise LeaderboardError(f"leaderboard {label} is invalid", code="leaderboard_schema")
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise LeaderboardError("leaderboard published_at is invalid", code="leaderboard_schema")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LeaderboardError("leaderboard published_at is invalid", code="leaderboard_schema") from exc
    if parsed.tzinfo is None:
        raise LeaderboardError("leaderboard published_at must include a timezone", code="leaderboard_schema")
    return value


def _gpu(value: Any) -> dict[str, str]:
    gpu = _exact_keys(value, PUBLIC_GPU_KEYS, "gpu")
    if gpu["vendor"] not in {"nvidia", "amd"}:
        raise LeaderboardError("leaderboard gpu vendor is invalid", code="leaderboard_schema")
    return {"vendor": gpu["vendor"], "model": _text(gpu["model"], "gpu model")}


def _entry(value: Any, view: str) -> dict[str, Any]:
    expected = PUBLIC_ENTRY_BASE_KEYS | (
        frozenset({"tokens_per_second", "sample_count"})
        if view == "throughput"
        else frozenset({"metric_id", "score", "higher_is_better"})
    )
    entry = _exact_keys(value, expected, f"{view} entry")
    rank = _integer(entry["rank"], "rank", minimum=1, maximum=MAX_TOTAL)
    entry_id = entry["entry_id"]
    if not isinstance(entry_id, str) or not ENTRY_ID.fullmatch(entry_id):
        raise LeaderboardError("leaderboard entry_id is invalid", code="leaderboard_schema")
    user_id = entry["user_id"]
    if not isinstance(user_id, str) or not USER_ID.fullmatch(user_id):
        raise LeaderboardError("leaderboard user_id is invalid", code="leaderboard_schema")
    model_id = _opaque(entry["model_id"], "model_id")
    revision = _opaque(entry["revision"], "revision")
    gpu = _gpu(entry["gpu"])
    published_at = _timestamp(entry["published_at"])
    if entry["view"] != view:
        raise LeaderboardError("leaderboard entry view does not match its collection", code="leaderboard_schema")
    result: dict[str, Any] = {
        "rank": rank,
        "user_id": user_id,
        "model_id": model_id,
        "revision": revision,
        "gpu": gpu,
        "published_at": published_at,
        "view": view,
    }
    if view == "throughput":
        result["tokens_per_second"] = _number(entry["tokens_per_second"], "tokens_per_second", maximum=MAX_THROUGHPUT)
        result["sample_count"] = _integer(entry["sample_count"], "sample_count", minimum=1, maximum=64)
    else:
        result["metric_id"] = _opaque(entry["metric_id"], "metric_id")
        result["score"] = _number(entry["score"])
        if not isinstance(entry["higher_is_better"], bool):
            raise LeaderboardError("leaderboard higher_is_better is invalid", code="leaderboard_schema")
        result["higher_is_better"] = entry["higher_is_better"]
    return result


def _configuration(value: Any, view: str) -> dict[str, Any]:
    group = _exact_keys(value, PUBLIC_CONFIGURATION_KEYS, f"{view} configuration")
    model_id = _opaque(group["model_id"], "configuration model_id")
    revision = _opaque(group["revision"], "configuration revision")
    gpu = _gpu(group["gpu"])
    total = _integer(group["total"], "configuration total", minimum=1, maximum=MAX_TOTAL)
    entries = group["entries"]
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES_PER_CONFIGURATION:
        raise LeaderboardError("leaderboard configuration entries are invalid", code="leaderboard_schema")
    safe_entries = [_entry(item, view) for item in entries]
    for entry in safe_entries:
        if (entry["model_id"], entry["revision"], entry["gpu"]) != (model_id, revision, gpu):
            raise LeaderboardError("leaderboard configuration contains a mismatched entry", code="leaderboard_schema")
    return {"model_id": model_id, "revision": revision, "gpu": gpu, "total": total, "entries": safe_entries}


def _view(value: Any, view: str) -> dict[str, Any]:
    data = _exact_keys(value, PUBLIC_VIEW_KEYS, f"{view} view")
    entries = data["entries"]
    if not isinstance(entries, list) or len(entries) > MAX_LIMIT:
        raise LeaderboardError("leaderboard entries are invalid", code="leaderboard_schema")
    total = _integer(data["total"], "total", minimum=0, maximum=MAX_TOTAL)
    offset = _integer(data["offset"], "offset", minimum=0, maximum=MAX_OFFSET)
    limit = _integer(data["limit"], "limit", minimum=1, maximum=MAX_LIMIT)
    safe_entries = [_entry(item, view) for item in entries]
    if any(entry["rank"] > max(total, 1) for entry in safe_entries):
        raise LeaderboardError("leaderboard rank is outside the result set", code="leaderboard_schema")
    configurations = data["configurations"]
    if not isinstance(configurations, list) or len(configurations) > MAX_CONFIGURATIONS:
        raise LeaderboardError("leaderboard configurations are invalid", code="leaderboard_schema")
    safe_configurations = [_configuration(item, view) for item in configurations]
    return {
        "entries": safe_entries,
        "total": total,
        "offset": offset,
        "limit": limit,
        "configurations": safe_configurations,
    }


def validate_public_leaderboard(value: Any) -> dict[str, Any]:
    """Validate the Hub envelope and return a private-field-free projection."""

    envelope = _exact_keys(value, frozenset({"ok", "schema_version", "views"}), "response")
    if envelope["ok"] is not True or envelope["schema_version"] != SCHEMA_VERSION:
        raise LeaderboardError("leaderboard response identity is invalid", code="leaderboard_schema")
    views = _object(envelope["views"], "views")
    if not views or any(view not in PUBLIC_VIEWS for view in views):
        raise LeaderboardError("leaderboard views are invalid", code="leaderboard_schema")
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "views": {view: _view(views[view], view) for view in sorted(views)},
    }


def _validate_endpoint(url: str) -> str:
    try:
        target = urlsplit(str(url or ""))
        hostname = target.hostname
        _ = target.port
    except ValueError as exc:
        raise LeaderboardError(
            "leaderboard URL must be HTTPS without credentials, query, or fragments",
            code="leaderboard_transport",
        ) from exc
    if (
        target.scheme.casefold() != "https"
        or not hostname
        or target.username
        or target.password
        or target.query
        or target.fragment
    ):
        raise LeaderboardError(
            "leaderboard URL must be HTTPS without credentials, query, or fragments",
            code="leaderboard_transport",
        )
    return url


def _query_value(value: str | None, label: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > MAX_TEXT or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise LeaderboardError(f"leaderboard {label} is invalid", code="leaderboard_query")
    return value


def _query_endpoint(url: str, *, view: str, model_id: str | None, revision: str | None,
                    gpu_vendor: str | None, gpu_model: str | None, search: str | None,
                    search_field: str, limit: int, offset: int) -> str:
    target = urlsplit(_validate_endpoint(url))
    params: dict[str, str | int] = {
        "view": view,
        "limit": limit,
        "offset": offset,
    }
    for key, value in (
        ("model_id", model_id), ("revision", revision), ("gpu_vendor", gpu_vendor),
        ("gpu_model", gpu_model), ("search", search), ("search_field", search_field),
    ):
        if value is not None and value != "":
            params[key] = value
    return urlunsplit((target.scheme, target.netloc, target.path, urlencode(params), ""))


def fetch_public_leaderboard(
    url: str,
    *,
    timeout: float = 10.0,
    view: str = "all",
    model_id: str | None = None,
    revision: str | None = None,
    gpu_vendor: str | None = None,
    gpu_model: str | None = None,
    search: str | None = None,
    search_field: str = "all",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """Fetch only the anonymous GET leaderboard; no bearer or write path exists here."""

    if view not in {"all", *PUBLIC_VIEWS}:
        raise LeaderboardError("leaderboard view is invalid", code="leaderboard_query")
    if gpu_vendor not in {None, "", "nvidia", "amd"}:
        raise LeaderboardError("leaderboard gpu_vendor is invalid", code="leaderboard_query")
    if search_field not in {"all", "model_id", "revision", "gpu_vendor", "gpu_model", "user_id", "metric_id"}:
        raise LeaderboardError("leaderboard search_field is invalid", code="leaderboard_query")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
        raise LeaderboardError("leaderboard limit is invalid", code="leaderboard_query")
    if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset <= MAX_OFFSET:
        raise LeaderboardError("leaderboard offset is invalid", code="leaderboard_query")
    for value, label, validator in (
        (model_id, "model_id", _opaque), (revision, "revision", _opaque),
        (gpu_model, "gpu_model", lambda item, name: _text(item, name)),
    ):
        if value is not None and value != "":
            validator(value, f"leaderboard {label}")
    safe_search = _query_value(search, "search")
    endpoint = _query_endpoint(
        url,
        view=view,
        model_id=model_id,
        revision=revision,
        gpu_vendor=gpu_vendor,
        gpu_model=gpu_model,
        search=safe_search,
        search_field=search_field,
        limit=limit,
        offset=offset,
    )
    request = urllib.request.Request(
        endpoint,
        headers={"Accept": "application/json", "User-Agent": "IDA_CLI-Public/0.2"},
        method="GET",
    )
    opener = urllib.request.build_opener(_HTTPSOriginRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise LeaderboardError("leaderboard response is too large", code="leaderboard_transport")
            if not 200 <= response.status < 300:
                raise LeaderboardError("leaderboard endpoint rejected the request", code="leaderboard_unavailable")
    except LeaderboardError:
        raise
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LeaderboardError("leaderboard request failed", code="leaderboard_unavailable") from exc
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeaderboardError("leaderboard response is not valid JSON", code="leaderboard_schema") from exc
    return validate_public_leaderboard(decoded)
