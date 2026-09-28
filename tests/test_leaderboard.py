from __future__ import annotations

import copy
import json
import urllib.request
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

import pytest

from ask_ida_cli.cli import main
from ask_ida_cli.leaderboard import (
    MAX_RESPONSE_BYTES,
    LeaderboardError,
    _HTTPSOriginRedirectHandler,
    fetch_public_leaderboard,
    validate_public_leaderboard,
)


def _payload() -> dict:
    entry = {
        "rank": 1,
        "entry_id": "nflb-12345678-1234-1234-1234-123456789abc",
        "user_id": "operator-123456789abc",
        "model_id": "ida-public",
        "revision": "v1",
        "gpu": {"vendor": "nvidia", "model": "RTX 5070"},
        "published_at": "2026-09-13T12:00:00.000Z",
        "view": "throughput",
        "tokens_per_second": 256.0,
        "sample_count": 1,
    }
    return {
        "ok": True,
        "schema_version": "neural-forge-leaderboard.v1",
        "views": {
            "throughput": {
                "entries": [entry],
                "total": 1,
                "offset": 0,
                "limit": 20,
                "configurations": [
                    {
                        "model_id": "ida-public",
                        "revision": "v1",
                        "gpu": {"vendor": "nvidia", "model": "RTX 5070"},
                        "total": 1,
                        "entries": [entry],
                    }
                ],
            }
        },
    }


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.body


class _Opener:
    def __init__(self, body: bytes):
        self.body = body
        self.request = None

    def open(self, request, timeout: float):
        self.request = request
        assert timeout == 10.0
        return _Response(self.body)


def test_valid_public_response_is_projected_without_entry_identity() -> None:
    result = validate_public_leaderboard(_payload())
    entry = result["views"]["throughput"]["entries"][0]
    assert entry["model_id"] == "ida-public"
    assert "entry_id" not in entry
    assert "run_id" not in entry
    assert "org" not in entry


def test_cli_leaderboard_is_an_explicit_read_only_get(monkeypatch, tmp_path, capsys) -> None:
    opener = _Opener(json.dumps(_payload()).encode("utf-8"))
    monkeypatch.setenv("ASK_IDA_PUBLIC_TELEMETRY_TOKEN", "must-not-be-forwarded")
    monkeypatch.setattr("ask_ida_cli.leaderboard.urllib.request.build_opener", lambda *_args: opener)

    assert main([
        "--root", str(tmp_path),
        "leaderboard", "list", "--url", "https://hidden-canopy-hub-api.azurewebsites.net/api/neural-forge/leaderboard",
        "--view", "throughput", "--search", "RTX 5070", "--json",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["views"]["throughput"]["entries"][0]["gpu"]["model"] == "RTX 5070"
    assert "entry_id" not in json.dumps(payload)
    assert opener.request.get_method() == "GET"
    assert opener.request.headers.get("Authorization") is None
    query = parse_qs(urlsplit(opener.request.full_url).query)
    assert query["view"] == ["throughput"]
    assert query["search"] == ["RTX 5070"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["views"]["throughput"]["entries"][0].update({"run_id": "nf-secret"}),
        lambda value: value["views"]["throughput"]["entries"][0].update({"published_at": "2026-09-13T12:00:00"}),
        lambda value: value["views"]["throughput"]["entries"][0].update({"gpu": {"vendor": "nvidia", "model": "../secret"}}),
        lambda value: value.update({"ok": False}),
    ],
)
def test_private_fallback_or_naive_data_fails_closed(mutate) -> None:
    candidate = copy.deepcopy(_payload())
    mutate(candidate)
    with pytest.raises(LeaderboardError):
        validate_public_leaderboard(candidate)


def test_transport_rejects_non_https_credentials_and_query() -> None:
    for url in (
        "http://example.invalid/leaderboard",
        "https://user:pass@example.invalid/leaderboard",
        "https://example.invalid/leaderboard?source=caller",
    ):
        with pytest.raises(LeaderboardError, match="HTTPS"):
            fetch_public_leaderboard(url)


def test_redirect_cannot_cross_https_origins() -> None:
    request = urllib.request.Request("https://hidden-canopy-hub-api.azurewebsites.net/api/neural-forge/leaderboard")
    handler = _HTTPSOriginRedirectHandler()
    with pytest.raises(LeaderboardError, match="original HTTPS origin"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://evil.example/leaderboard")


def test_oversized_response_is_rejected(monkeypatch) -> None:
    opener = _Opener(b"x" * (MAX_RESPONSE_BYTES + 1))
    monkeypatch.setattr("ask_ida_cli.leaderboard.urllib.request.build_opener", lambda *_args: opener)
    with pytest.raises(LeaderboardError, match="too large"):
        fetch_public_leaderboard("https://example.invalid/leaderboard")


def test_query_bounds_are_checked_before_network() -> None:
    with pytest.raises(LeaderboardError, match="limit"):
        fetch_public_leaderboard("https://example.invalid/leaderboard", limit=101)
    with pytest.raises(LeaderboardError, match="offset"):
        fetch_public_leaderboard("https://example.invalid/leaderboard", offset=100001)
