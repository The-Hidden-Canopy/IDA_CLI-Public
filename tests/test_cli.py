from __future__ import annotations

import json
from pathlib import Path

from ask_ida_cli.cli import main


def test_status_is_local_and_does_not_need_catalog(tmp_path: Path, capsys) -> None:
    assert main(["--root", str(tmp_path), "status", "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["network"] == "explicit_catalog_download_leaderboard_or_telemetry_only"
    assert payload["private_runtime"] == "not_available_in_public_cli"


def test_public_documentation_explanation_is_bundled(capsys) -> None:
    assert main(["explain", "private", "boundary", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_state"] == "bundled_public_docs"
    assert payload["results"]
    assert all(result["source"] == "public-cli-contract" for result in payload["results"])


def test_model_download_requires_explicit_id(capsys) -> None:
    assert main(["model", "download", "--json"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "model_id_required"


def test_inference_path_cannot_escape_public_model_root(tmp_path: Path, capsys) -> None:
    outside = tmp_path / "outside-model"
    assert main([
        "--root", str(tmp_path),
        "--model-root", str(tmp_path / "models"),
        "ask", str(outside), "hello", "there", "--json",
    ]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "model_path"


def test_inference_accepts_a_local_model_directory_name_without_network(tmp_path: Path, monkeypatch, capsys) -> None:
    model_root = tmp_path / "models"
    model = model_root / "public-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("ask_ida_cli.cli.ask_local", lambda path, prompt, max_new_tokens: f"{path.name}:{prompt}")

    assert main([
        "--root", str(tmp_path),
        "--model-root", str(model_root),
        "ask", "public-model", "hello", "there", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["response"] == "public-model:hello there"
