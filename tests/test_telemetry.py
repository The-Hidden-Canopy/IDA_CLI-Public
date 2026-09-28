from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from ask_ida_cli.cli import main
from ask_ida_cli.telemetry import (
    MAX_FILE_BYTES,
    TelemetryError,
    _HTTPSOnlyRedirectHandler,
    load_share_bundle,
    share_kernel_telemetry,
)


def _write_bundle(root: Path) -> Path:
    root.mkdir()
    ontology_rows = [
        {
            "type": "context",
            "telemetry_schema": "native_ontology_v2",
            "ctx": 0,
            "microbatch": 1,
            "grad_accum": 1,
            "num_experts": 0,
            "top_k_experts": 0,
            "seq_window": 128,
            "precision_profile": "bf16",
            "attention_backend": "scalar_flash",
            "optimizer": "lion",
            "kernel_collection": "public_bounded",
        },
        {
            "type": "launch",
            "telemetry_schema": "native_ontology_v2",
            "ctx": 0,
            "cuda_device": 0,
            "kernel": "k_add_bf16",
            "role": "attention.residual_add",
            "grid": [1, 1, 1],
            "block": [256, 1, 1],
            "blocks": 1,
            "threads_per_block": 256,
            "warp_fill": 1.0,
            "dyn_smem": 0,
            "static_smem": 0,
            "regs_per_thread": 32,
            "local_bytes": 0,
            "max_threads_per_block": 1024,
            "sm_count": 132,
            "findings": ["SM_STARVED"],
        },
        {
            "type": "objective",
            "telemetry_schema": "native_ontology_v2",
            "ctx": 0,
            "opt_step": 1,
            "loss": 1.25,
            "wallclock_s": 0.5,
            "tokens": 128,
            "update_skipped": False,
            "clip_fired": False,
            "grad_norm": 0.75,
            "timing_scope": "host_observation",
            "device_execution_timing": False,
        },
    ]
    metrics_rows = [
        {
            "record_type": "native_kernel_telemetry_v2",
            "telemetry_schema": "native_kernel_telemetry_v2",
            "backend": "native",
            "global_step": 1,
            "optimizer_steps": 1,
            "tokens_processed": 128,
            "elapsed_seconds": 0.5,
            "tokens_per_second": 256.0,
            "promotion_eligible": False,
            "profile_claim": "native_ontology_telemetry_only",
            "coverage_scope": "wrapped_public_launch_sites_only",
            "hardware_counter_profile": False,
            "device_elapsed_kernel_time": False,
        }
    ]
    ontology_path = root / "kernel_ontology.jsonl"
    metrics_path = root / "training_metrics.jsonl"
    ontology_path.write_text("".join(json.dumps(row) + "\n" for row in ontology_rows), encoding="utf-8")
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in metrics_rows), encoding="utf-8")
    receipt = {
        "schema_version": "native-kernel-telemetry.v2",
        "source": "canopy-foundry",
        "claims": {
            "profile_claim": "native_ontology_telemetry_only",
            "coverage_scope": "wrapped_public_launch_sites_only",
            "external_profiler_used": False,
            "hardware_counter_profile": False,
            "device_elapsed_kernel_time": False,
            "promotion_eligible": False,
        },
        "counts": {
            "kernel_ontology_records": len(ontology_rows),
            "training_metric_records": len(metrics_rows),
        },
        "files": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in {
                "kernel_ontology.jsonl": ontology_path,
                "training_metrics.jsonl": metrics_path,
            }.items()
        },
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return root


def _refresh_receipt_file_hash(bundle: Path, filename: str) -> None:
    path = bundle / filename
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"][filename] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_kernel_telemetry_status_is_disabled_by_default(tmp_path: Path, capsys) -> None:
    assert main(["--root", str(tmp_path), "telemetry", "status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sharing"] == "disabled_by_default"
    assert payload["schema_version"] == "native-kernel-telemetry.v2"
    assert payload["scope"] == "kernel_ontology_and_training_metrics"


def test_kernel_telemetry_dry_run_reconciles_receipt_and_streams(tmp_path: Path, capsys) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    assert main(["telemetry", "share-kernel", str(bundle), "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "validated"
    assert payload["bundle"]["kernel_ontology_records"] == 3
    assert payload["bundle"]["training_metric_records"] == 1


def test_kernel_telemetry_rejects_tampering(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    (bundle / "kernel_ontology.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        load_share_bundle(bundle)
    except Exception as exc:  # noqa: BLE001 - integrity must fail closed
        assert "hash" in str(exc)
    else:
        raise AssertionError("tampered kernel telemetry must be rejected")


def test_kernel_telemetry_rejects_private_row_fields_even_with_recomputed_hash(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    path = bundle / "kernel_ontology.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["dataset_path"] = "C:/private/dataset"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"]["kernel_ontology.jsonl"] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    try:
        load_share_bundle(bundle)
    except Exception as exc:  # noqa: BLE001 - private fields must fail closed
        assert "unknown field" in str(exc)
    else:
        raise AssertionError("private kernel telemetry fields must be rejected")


def test_kernel_telemetry_rejects_type_confused_launch_geometry(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    path = bundle / "kernel_ontology.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["grid"] = "not-a-grid"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _refresh_receipt_file_hash(bundle, "kernel_ontology.jsonl")
    with pytest.raises(TelemetryError, match="grid"):
        load_share_bundle(bundle)


def test_kernel_telemetry_rejects_string_numeric_metric(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    path = bundle / "training_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["global_step"] = "1"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _refresh_receipt_file_hash(bundle, "training_metrics.jsonl")
    with pytest.raises(TelemetryError, match="global_step"):
        load_share_bundle(bundle)


def test_kernel_telemetry_rejects_extra_receipt_count_fields(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["counts"]["unexpected_private_field"] = "secret"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(TelemetryError, match="counts are invalid"):
        load_share_bundle(bundle)


def test_kernel_telemetry_rejects_path_shaped_row_values(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    path = bundle / "kernel_ontology.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["kernel"] = "C:/private/checkpoint"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"]["kernel_ontology.jsonl"] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(TelemetryError, match="telemetry kernel is invalid"):
        load_share_bundle(bundle)


def test_kernel_telemetry_rejects_oversized_input_before_hashing(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    (bundle / "kernel_ontology.jsonl").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    with pytest.raises(TelemetryError, match="file is too large"):
        load_share_bundle(bundle)


def test_kernel_telemetry_rejects_symlinked_bundle_member(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    target = tmp_path / "outside.jsonl"
    target.write_bytes((bundle / "kernel_ontology.jsonl").read_bytes())
    member = bundle / "kernel_ontology.jsonl"
    member.unlink()
    try:
        os.symlink(target, member)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(TelemetryError, match="must not be a link"):
        load_share_bundle(bundle)


def test_kernel_telemetry_rejects_symlinked_bundle_root(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    alias = tmp_path / "kernel-telemetry-alias"
    try:
        os.symlink(bundle, alias, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(TelemetryError, match="must not pass through a link"):
        load_share_bundle(alias)


def test_kernel_telemetry_share_requires_an_explicit_endpoint(tmp_path: Path, capsys) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    assert main(["telemetry", "share-kernel", str(bundle), "--json"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "telemetry_url_required"


def test_kernel_telemetry_token_requires_an_explicit_endpoint(tmp_path: Path, capsys, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    monkeypatch.setenv("ASK_IDA_PUBLIC_TELEMETRY_TOKEN", "secret")
    monkeypatch.setenv("ASK_IDA_PUBLIC_TELEMETRY_URL", "https://configured.example.invalid/ingest")
    assert main(["telemetry", "share-kernel", str(bundle), "--send-token", "--json"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "telemetry_token_requires_url"


def test_kernel_telemetry_share_is_the_explicit_cli_transport(tmp_path: Path, capsys, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    captured: dict[str, object] = {}
    monkeypatch.delenv("ASK_IDA_PUBLIC_TELEMETRY_TOKEN", raising=False)

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit: int) -> bytes:
            captured["response_limit"] = limit
            return b"{}"

    class FakeOpener:
        def open(self, request, timeout: float):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

    with patch("ask_ida_cli.telemetry.urllib.request.build_opener", return_value=FakeOpener()):
        assert main(
            [
                "--root",
                str(tmp_path),
                "telemetry",
                "share-kernel",
                str(bundle),
                "--url",
                "https://telemetry.example.invalid/ingest",
                "--json",
            ]
        ) == 0

    result = json.loads(capsys.readouterr().out)
    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert result["status"] == "shared"
    assert request.get_method() == "POST"
    assert request.get_header("User-agent") == "IDA_CLI-Public/0.2"
    assert captured["timeout"] == 10.0
    assert captured["response_limit"] == 64 * 1024 + 1
    assert set(payload) == {"schema_version", "source", "receipt", "kernel_ontology", "training_metrics"}
    assert payload["schema_version"] == "native-kernel-telemetry.v2"
    assert payload["source"] == "canopy-foundry"


def test_kernel_telemetry_token_is_not_sent_without_the_explicit_flag(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    captured: dict[str, object] = {}
    monkeypatch.setenv("ASK_IDA_PUBLIC_TELEMETRY_TOKEN", "secret")

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit: int) -> bytes:
            return b"{}"

    class FakeOpener:
        def open(self, request, timeout: float):
            captured["request"] = request
            return FakeResponse()

    with patch("ask_ida_cli.telemetry.urllib.request.build_opener", return_value=FakeOpener()):
        result = share_kernel_telemetry(
            bundle,
            "https://telemetry.example.invalid/ingest",
            timeout=10.0,
        )
    assert result["status"] == "shared"
    assert captured["request"].get_header("Authorization") is None


def test_kernel_telemetry_token_requires_an_environment_value(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    monkeypatch.delenv("ASK_IDA_PUBLIC_TELEMETRY_TOKEN", raising=False)
    with pytest.raises(TelemetryError, match="valid ASK_IDA_PUBLIC_TELEMETRY_TOKEN"):
        share_kernel_telemetry(
            bundle,
            "https://telemetry.example.invalid/ingest",
            timeout=10.0,
            send_token=True,
        )


def test_kernel_telemetry_token_is_sent_only_when_requested(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")
    captured: dict[str, object] = {}
    monkeypatch.setenv("ASK_IDA_PUBLIC_TELEMETRY_TOKEN", "secret")

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit: int) -> bytes:
            return b"{}"

    class FakeOpener:
        def open(self, request, timeout: float):
            captured["request"] = request
            return FakeResponse()

    with patch("ask_ida_cli.telemetry.urllib.request.build_opener", return_value=FakeOpener()):
        share_kernel_telemetry(
            bundle,
            "https://telemetry.example.invalid/ingest",
            timeout=10.0,
            send_token=True,
        )
    assert captured["request"].get_header("Authorization") == "Bearer secret"



def test_kernel_telemetry_redirect_cannot_cross_https_origins() -> None:
    request = urllib.request.Request("https://telemetry.example.invalid/ingest")
    try:
        _HTTPSOnlyRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://other.example.invalid/collect",
        )
    except TelemetryError as exc:
        assert "original HTTPS origin" in str(exc)
    else:
        raise AssertionError("telemetry redirects must not cross HTTPS origins")


def test_kernel_telemetry_share_rejects_an_oversized_response(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "kernel-telemetry")

    class OversizedResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit: int) -> bytes:
            return b"x" * limit

    class OversizedOpener:
        def open(self, request, timeout: float):
            return OversizedResponse()

    monkeypatch.setattr(
        "ask_ida_cli.telemetry.urllib.request.build_opener",
        lambda *args: OversizedOpener(),
    )
    try:
        share_kernel_telemetry(bundle, "https://telemetry.example.invalid/ingest", timeout=10.0)
    except TelemetryError as exc:
        assert "response is too large" in str(exc)
    else:
        raise AssertionError("oversized telemetry responses must be rejected")
