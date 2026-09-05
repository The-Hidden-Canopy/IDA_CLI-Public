from __future__ import annotations

from pathlib import Path

import pytest

from ask_ida_cli.catalog import CatalogError, content_manifest
from ask_ida_cli.models import download_experimental, download_reviewed, list_local_models


def _catalog(digest: str) -> dict:
    return {
        "public_hosts": ["huggingface.co"],
        "models": [{
            "artifact_id": "fixture-model",
            "kind": "model",
            "status": "supported",
            "artifact_sha256": digest,
            "source": {"host": "huggingface.co", "repo_id": "org/model", "revision": "0123456789abcdef"},
        }],
    }


def _write_fixture(root: Path) -> None:
    (root / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"fixture-weights")


def test_reviewed_download_verifies_digest_and_commits_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    _write_fixture(expected_dir)
    digest = content_manifest(expected_dir)["sha256"]

    def fake_download(**kwargs):
        _write_fixture(Path(kwargs["local_dir"]))
        return kwargs["local_dir"]

    monkeypatch.setattr("ask_ida_cli.models._snapshot_download", lambda: fake_download)
    result = download_reviewed(_catalog(digest), "fixture-model", tmp_path / "models")
    assert result["status"] == "verified"
    assert (tmp_path / "models" / "fixture-model" / "config.json").is_file()


def test_reviewed_download_rejects_digest_drift_and_leaves_no_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(**kwargs):
        _write_fixture(Path(kwargs["local_dir"]))
        return kwargs["local_dir"]

    monkeypatch.setattr("ask_ida_cli.models._snapshot_download", lambda: fake_download)
    with pytest.raises(CatalogError, match="does not match"):
        download_reviewed(_catalog("b" * 64), "fixture-model", tmp_path / "models")
    assert not (tmp_path / "models" / "fixture-model").exists()


def test_experimental_download_marks_unsupported_but_rejects_remote_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(**kwargs):
        target = Path(kwargs["local_dir"])
        _write_fixture(target)
        (target / "modeling_custom.py").write_text("raise RuntimeError('must not run')", encoding="utf-8")
        return kwargs["local_dir"]

    monkeypatch.setattr("ask_ida_cli.models._snapshot_download", lambda: fake_download)
    with pytest.raises(CatalogError, match="disallowed"):
        download_experimental("org/model", tmp_path / "models")
    assert not (tmp_path / "models" / "org--model").exists()


def test_local_listing_is_truthful_for_complete_models(tmp_path: Path) -> None:
    model = tmp_path / "model-a"
    model.mkdir()
    _write_fixture(model)
    rows = list_local_models(tmp_path)
    assert rows[0]["directory"] == "model-a"
    assert rows[0]["ready"] is True
    assert rows[0]["artifact_sha256"]

