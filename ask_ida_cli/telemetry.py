"""Validate and explicitly share the bounded Foundry kernel telemetry bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

from .errors import CLIError

SCHEMA_VERSION = "native-kernel-telemetry.v2"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_RECORDS = 250_000
REQUIRED_FILES = ("kernel_ontology.jsonl", "training_metrics.jsonl")
RECEIPT_KEYS = frozenset({"schema_version", "source", "claims", "counts", "files"})
COUNT_KEYS = frozenset({"kernel_ontology_records", "training_metric_records"})
CLAIM_KEYS = frozenset({
    "profile_claim",
    "coverage_scope",
    "external_profiler_used",
    "hardware_counter_profile",
    "device_elapsed_kernel_time",
    "promotion_eligible",
})
FILE_KEYS = frozenset({"bytes", "sha256"})
ONTOLOGY_KEYS = {
    "context": frozenset({
        "type", "telemetry_schema", "ctx", "microbatch", "grad_accum",
        "num_experts", "top_k_experts", "seq_window", "precision_profile",
        "attention_backend", "optimizer", "kernel_collection",
    }),
    "launch": frozenset({
        "type", "telemetry_schema", "ctx", "cuda_device", "kernel", "role",
        "grid", "block", "blocks", "threads_per_block", "warp_fill", "dyn_smem",
        "static_smem", "regs_per_thread", "local_bytes", "max_threads_per_block",
        "sm_count", "findings",
    }),
    "launch_timing": frozenset({
        "type", "telemetry_schema", "ctx", "cuda_device", "opt_step", "kernel",
        "role", "grid", "block", "dyn_smem", "launches", "enqueue_ns_sum",
        "enqueue_us_sum", "enqueue_us_mean", "enqueue_us_min", "enqueue_us_max",
        "timing_scope", "device_execution_timing",
    }),
    "objective": frozenset({
        "type", "telemetry_schema", "ctx", "opt_step", "loss", "wallclock_s",
        "tokens", "update_skipped", "clip_fired", "grad_norm", "timing_scope",
        "device_execution_timing",
    }),
}
METRIC_KEYS = frozenset({
    "record_type", "telemetry_schema", "backend", "global_step", "optimizer_steps",
    "tokens_processed", "elapsed_seconds", "tokens_per_second", "promotion_eligible",
    "profile_claim", "coverage_scope", "hardware_counter_profile", "device_elapsed_kernel_time",
})
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.()+-]{1,128}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TelemetryError(CLIError):
    """A kernel telemetry bundle or transport is invalid."""


def _number(value: Any, label: str, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryError(f"telemetry {label} is invalid", code="telemetry_schema")
    if isinstance(value, float) and not math.isfinite(value):
        raise TelemetryError(f"telemetry {label} is invalid", code="telemetry_schema")
    if value < -(10**18) or value > 10**18:
        raise TelemetryError(f"telemetry {label} is invalid", code="telemetry_schema")
    if integer and not isinstance(value, int):
        raise TelemetryError(f"telemetry {label} must be an integer", code="telemetry_schema")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise TelemetryError(f"telemetry {label} is invalid", code="telemetry_schema")
    if ".." in value or "\\" in value:
        raise TelemetryError(f"telemetry {label} is invalid", code="telemetry_schema")
    return value


def _required_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    result = _number(value, label, integer=True)
    if result is None:
        raise TelemetryError(f"telemetry {label} is required", code="telemetry_schema")
    if minimum is not None and result < minimum:
        raise TelemetryError(f"telemetry {label} is out of range", code="telemetry_schema")
    return result


def _required_number(value: Any, label: str, *, minimum: int | float | None = None) -> int | float:
    result = _number(value, label)
    if result is None:
        raise TelemetryError(f"telemetry {label} is required", code="telemetry_schema")
    if minimum is not None and result < minimum:
        raise TelemetryError(f"telemetry {label} is out of range", code="telemetry_schema")
    return result


def _required_boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TelemetryError(f"telemetry {label} must be a boolean", code="telemetry_schema")
    return value


def _required_vector(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise TelemetryError(f"telemetry {label} must be a 3-element integer vector", code="telemetry_schema")
    for index, item in enumerate(value):
        _required_integer(item, f"{label}[{index}]", minimum=0)


def _required_token_list(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise TelemetryError(f"telemetry {label} must be a list", code="telemetry_schema")
    for index, item in enumerate(value):
        _token(item, f"{label}[{index}]")


def _validate_row_types(row: dict[str, Any], *, metric: bool) -> None:
    if metric:
        if row["record_type"] != "native_kernel_telemetry_v2":
            raise TelemetryError("telemetry record type is invalid", code="telemetry_schema")
        for key in ("global_step", "optimizer_steps", "tokens_processed"):
            _required_integer(row[key], key, minimum=0)
        for key in ("elapsed_seconds", "tokens_per_second"):
            _required_number(row[key], key, minimum=0)
        _required_boolean(row["promotion_eligible"], "promotion_eligible")
        _required_boolean(row["hardware_counter_profile"], "hardware_counter_profile")
        _required_boolean(row["device_elapsed_kernel_time"], "device_elapsed_kernel_time")
        if row["profile_claim"] != "native_ontology_telemetry_only":
            raise TelemetryError("telemetry profile claim is invalid", code="telemetry_schema")
        return

    row_type = row["type"]
    _required_integer(row["ctx"], "ctx", minimum=0)
    if row_type == "context":
        for key in ("microbatch", "grad_accum", "num_experts", "top_k_experts", "seq_window"):
            _required_integer(row[key], key)
        if row["kernel_collection"] != "public_bounded":
            raise TelemetryError("telemetry kernel collection is invalid", code="telemetry_schema")
        return
    if row_type == "launch":
        _required_integer(row["cuda_device"], "cuda_device", minimum=-1)
        _required_vector(row["grid"], "grid")
        _required_vector(row["block"], "block")
        for key in (
            "blocks", "threads_per_block", "dyn_smem", "static_smem", "regs_per_thread",
            "local_bytes", "max_threads_per_block", "sm_count",
        ):
            _required_integer(row[key], key, minimum=0)
        _required_number(row["warp_fill"], "warp_fill", minimum=0)
        _required_token_list(row["findings"], "findings")
        return
    if row_type == "launch_timing":
        _required_integer(row["cuda_device"], "cuda_device", minimum=-1)
        _required_integer(row["opt_step"], "opt_step")
        _required_vector(row["grid"], "grid")
        _required_vector(row["block"], "block")
        for key in ("dyn_smem", "launches", "enqueue_ns_sum"):
            _required_integer(row[key], key, minimum=0)
        for key in ("enqueue_us_sum", "enqueue_us_mean", "enqueue_us_min", "enqueue_us_max"):
            _required_number(row[key], key, minimum=0)
        if row["timing_scope"] != "host_enqueue_only":
            raise TelemetryError("telemetry timing scope is invalid", code="telemetry_schema")
        _required_boolean(row["device_execution_timing"], "device_execution_timing")
        return
    if row_type == "objective":
        _required_integer(row["opt_step"], "opt_step")
        _required_number(row["loss"], "loss")
        _required_number(row["wallclock_s"], "wallclock_s", minimum=0)
        _required_integer(row["tokens"], "tokens", minimum=0)
        _required_boolean(row["update_skipped"], "update_skipped")
        _required_boolean(row["clip_fired"], "clip_fired")
        _required_number(row["grad_norm"], "grad_norm", minimum=0)
        if row["timing_scope"] != "host_observation":
            raise TelemetryError("telemetry timing scope is invalid", code="telemetry_schema")
        _required_boolean(row["device_execution_timing"], "device_execution_timing")


def _row_shape(row: dict[str, Any], *, metric: bool) -> None:
    row_type = row.get("record_type") if metric else row.get("type")
    if not isinstance(row_type, str):
        raise TelemetryError("telemetry record type is missing", code="telemetry_schema")
    allowed = METRIC_KEYS if metric else ONTOLOGY_KEYS.get(row_type)
    if allowed is None or set(row) != allowed:
        raise TelemetryError("telemetry record contains an unknown field", code="telemetry_schema")
    schema = row.get("telemetry_schema")
    if schema not in {"native_ontology_v2", "native_ontology_v3", "native_kernel_telemetry_v2", SCHEMA_VERSION}:
        raise TelemetryError("telemetry record schema is invalid", code="telemetry_schema")
    for key, value in row.items():
        if isinstance(value, str):
            _token(value, key)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)) or value is None:
            _number(value, key)
        elif isinstance(value, list):
            if len(value) > 16:
                raise TelemetryError("telemetry array is too large", code="telemetry_schema")
            for item in value:
                if isinstance(item, str):
                    _token(item, key)
                else:
                    _number(item, key)
        else:
            raise TelemetryError("telemetry record value is invalid", code="telemetry_schema")
    if metric:
        if row.get("backend") != "native" or row.get("promotion_eligible") is not False:
            raise TelemetryError("shared telemetry cannot claim promotion eligibility", code="telemetry_schema")
        if row.get("coverage_scope") != "wrapped_public_launch_sites_only":
            raise TelemetryError("kernel telemetry coverage scope is invalid", code="telemetry_schema")
    _validate_row_types(row, metric=metric)


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _reject_linked_components(path: Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if _is_link(component):
            raise TelemetryError(f"kernel telemetry {label} must not pass through a link", code="telemetry_input")


def _regular_file_size(path: Path, *, label: str) -> int:
    if _is_link(path):
        raise TelemetryError(f"kernel telemetry {label} must not be a link", code="telemetry_input")
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise TelemetryError(f"kernel telemetry {label} is missing", code="telemetry_input") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise TelemetryError(f"kernel telemetry {label} must be a regular file", code="telemetry_input")
    if info.st_size < 0:
        raise TelemetryError(f"kernel telemetry {label} size is invalid", code="telemetry_integrity")
    return int(info.st_size)


def _read_bounded_file(path: Path, *, maximum: int, label: str) -> bytes:
    """Read one stable, regular, non-link file without exceeding its byte cap."""

    _regular_file_size(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            before = os.fstat(stream.fileno())
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise TelemetryError(f"kernel telemetry {label} must be a regular file", code="telemetry_input")
            if before.st_size > maximum:
                raise TelemetryError(f"kernel telemetry {label} is too large", code="telemetry_size")
            raw = stream.read(maximum + 1)
            if len(raw) > maximum:
                raise TelemetryError(f"kernel telemetry {label} is too large", code="telemetry_size")
            if len(raw) != before.st_size:
                raise TelemetryError(f"kernel telemetry {label} changed while reading", code="telemetry_integrity")
        after = os.lstat(path)
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or getattr(after, "st_dev", None) != getattr(before, "st_dev", None)
            or getattr(after, "st_ino", None) != getattr(before, "st_ino", None)
            or after.st_size != before.st_size
            or getattr(after, "st_mtime_ns", None) != getattr(before, "st_mtime_ns", None)
        ):
            raise TelemetryError(f"kernel telemetry {label} changed while reading", code="telemetry_integrity")
        return raw
    except TelemetryError:
        raise
    except MemoryError as exc:
        raise TelemetryError(
            f"kernel telemetry {label} could not be held in memory",
            code="telemetry_size",
        ) from exc
    except (OSError, ValueError) as exc:
        raise TelemetryError(f"unable to read kernel telemetry {label}", code="telemetry_input") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _parse_jsonl(raw: bytes, *, metric: bool) -> list[dict[str, Any]]:
    if len(raw) > MAX_FILE_BYTES:
        raise TelemetryError("kernel telemetry file is too large", code="telemetry_size")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TelemetryError("kernel telemetry must be UTF-8", code="telemetry_json") from exc
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TelemetryError("kernel telemetry contains invalid JSON", code="telemetry_json") from exc
        if not isinstance(row, dict):
            raise TelemetryError("kernel telemetry rows must be objects", code="telemetry_schema")
        _row_shape(row, metric=metric)
        rows.append(row)
        if len(rows) > MAX_RECORDS:
            raise TelemetryError("kernel telemetry contains too many records", code="telemetry_size")
    if not rows:
        raise TelemetryError("kernel telemetry stream is empty", code="telemetry_schema")
    return rows


def _read_jsonl(path: Path, *, metric: bool) -> list[dict[str, Any]]:
    return _parse_jsonl(
        _read_bounded_file(path, maximum=MAX_FILE_BYTES, label=path.name),
        metric=metric,
    )


def _read_bundle(path: str | Path) -> dict[str, Any]:
    input_root = Path(path).expanduser()
    _reject_linked_components(input_root, label="input directory")
    try:
        root = input_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TelemetryError("kernel telemetry input directory is invalid", code="telemetry_input") from exc
    if _is_link(root):
        raise TelemetryError("kernel telemetry input directory must not be a link", code="telemetry_input")
    if not root.is_dir():
        raise TelemetryError("kernel telemetry input must be a directory", code="telemetry_input")
    receipt_path = root / "receipt.json"
    try:
        receipt = json.loads(
            _read_bounded_file(receipt_path, maximum=256 * 1024, label="receipt").decode("utf-8")
        )
    except TelemetryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TelemetryError("kernel telemetry receipt is invalid", code="telemetry_json") from exc
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise TelemetryError("kernel telemetry receipt shape is invalid", code="telemetry_schema")
    if receipt["schema_version"] != SCHEMA_VERSION or receipt["source"] != "canopy-foundry":
        raise TelemetryError("kernel telemetry receipt identity is invalid", code="telemetry_schema")
    claims = receipt["claims"]
    if not isinstance(claims, dict) or set(claims) != CLAIM_KEYS:
        raise TelemetryError("kernel telemetry claims are invalid", code="telemetry_schema")
    if (
        claims["profile_claim"] != "native_ontology_telemetry_only"
        or claims["coverage_scope"] != "wrapped_public_launch_sites_only"
        or claims["external_profiler_used"] is not False
        or claims["hardware_counter_profile"] is not False
        or claims["device_elapsed_kernel_time"] is not False
        or claims["promotion_eligible"] is not False
    ):
        raise TelemetryError("kernel telemetry claims are not bounded", code="telemetry_schema")
    counts = receipt["counts"]
    if not isinstance(counts, dict) or set(counts) != COUNT_KEYS:
        raise TelemetryError("kernel telemetry receipt counts are invalid", code="telemetry_schema")
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_RECORDS:
            raise TelemetryError(f"kernel telemetry count {key} is invalid", code="telemetry_schema")
    files = receipt["files"]
    if not isinstance(files, dict) or set(files) != set(REQUIRED_FILES):
        raise TelemetryError("kernel telemetry file manifest is invalid", code="telemetry_schema")
    file_paths: dict[str, Path] = {}
    file_sizes: dict[str, int] = {}
    total_bytes = 0
    for name in REQUIRED_FILES:
        entry = files[name]
        if not isinstance(entry, dict) or set(entry) != FILE_KEYS:
            raise TelemetryError("kernel telemetry file entry is invalid", code="telemetry_schema")
        if (
            isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or not SHA256.fullmatch(entry["sha256"])
        ):
            raise TelemetryError("kernel telemetry file entry is invalid", code="telemetry_schema")
        file_path = root / name
        size = _regular_file_size(file_path, label=name)
        if size > MAX_FILE_BYTES:
            raise TelemetryError("kernel telemetry file is too large", code="telemetry_size")
        file_paths[name] = file_path
        file_sizes[name] = size
        total_bytes += size
        if total_bytes > MAX_TOTAL_BYTES:
            raise TelemetryError("kernel telemetry bundle is too large", code="telemetry_size")
    raw_files = {
        name: _read_bounded_file(file_paths[name], maximum=MAX_FILE_BYTES, label=name)
        for name in REQUIRED_FILES
    }
    for name in REQUIRED_FILES:
        if len(raw_files[name]) != file_sizes[name]:
            raise TelemetryError(f"kernel telemetry {name} changed while reading", code="telemetry_integrity")
        digest = hashlib.sha256(raw_files[name]).hexdigest()
        entry = files[name]
        if entry["bytes"] != file_sizes[name] or entry["sha256"] != digest:
            raise TelemetryError("kernel telemetry receipt hash mismatch", code="telemetry_integrity")
    ontology = _parse_jsonl(raw_files["kernel_ontology.jsonl"], metric=False)
    metrics = _parse_jsonl(raw_files["training_metrics.jsonl"], metric=True)
    del raw_files
    if counts["kernel_ontology_records"] != len(ontology) or counts["training_metric_records"] != len(metrics):
        raise TelemetryError("kernel telemetry receipt counts do not reconcile", code="telemetry_integrity")
    return {
        "root": root,
        "receipt": receipt,
        "ontology": ontology,
        "metrics": metrics,
        "bytes": total_bytes,
    }


def load_share_bundle(path: str | Path) -> dict[str, Any]:
    """Validate a Foundry V2-style kernel telemetry directory and return a summary."""

    bundle = _read_bundle(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "canopy-foundry",
        "files": list(REQUIRED_FILES),
        "kernel_ontology_records": len(bundle["ontology"]),
        "training_metric_records": len(bundle["metrics"]),
        "bytes": bundle["bytes"],
        "claims": bundle["receipt"]["claims"],
    }


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        try:
            source = urlsplit(req.full_url)
            target = urlsplit(newurl)
            source_host = source.hostname
            target_host = target.hostname
            source_port = source.port or 443
            target_port = target.port or 443
        except ValueError as exc:
            raise TelemetryError("telemetry redirect URL is invalid", code="telemetry_transport") from exc
        if (
            source.scheme.casefold() != "https"
            or target.scheme.casefold() != "https"
            or not source_host
            or not target_host
            or source_host.casefold() != target_host.casefold()
            or source_port != target_port
            or target.username
            or target.password
            or target.query
            or target.fragment
        ):
            raise TelemetryError(
                "telemetry redirect must remain on the original HTTPS origin",
                code="telemetry_transport",
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_endpoint(url: str) -> str:
    try:
        target = urlsplit(str(url or ""))
        hostname = target.hostname
        target.port
    except ValueError as exc:
        raise TelemetryError(
            "telemetry URL must be HTTPS without credentials, query, or fragment",
            code="telemetry_transport",
        ) from exc
    if (
        target.scheme.casefold() != "https"
        or not hostname
        or target.username
        or target.password
        or target.query
        or target.fragment
    ):
        raise TelemetryError(
            "telemetry URL must be HTTPS without credentials, query, or fragment",
            code="telemetry_transport",
        )
    return url


def share_kernel_telemetry(
    path: str | Path,
    url: str,
    *,
    timeout: float,
    send_token: bool = False,
) -> dict[str, Any]:
    bundle = _read_bundle(path)
    target = _validate_endpoint(url)
    token = os.environ.get("ASK_IDA_PUBLIC_TELEMETRY_TOKEN") if send_token else None
    if send_token and (
        not token or len(token) > 4096 or any(character in token for character in "\r\n")
    ):
        raise TelemetryError(
            "--send-token requires a valid ASK_IDA_PUBLIC_TELEMETRY_TOKEN",
            code="telemetry_token_required",
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "canopy-foundry",
        "receipt": bundle["receipt"],
        "kernel_ontology": bundle["ontology"],
        "training_metrics": bundle["metrics"],
    }
    request = urllib.request.Request(
        target,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "IDA_CLI-Public/0.2",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    opener = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            response_body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(response_body) > MAX_RESPONSE_BYTES:
                raise TelemetryError("telemetry endpoint response is too large", code="telemetry_transport")
            if not 200 <= response.status < 300:
                raise TelemetryError("telemetry endpoint rejected the bundle", code="telemetry_rejected")
    except TelemetryError:
        raise
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TelemetryError("telemetry share request failed", code="telemetry_unavailable") from exc
    return {
        "status": "shared",
        "schema_version": SCHEMA_VERSION,
        "endpoint_host": urlsplit(target).hostname,
        "kernel_ontology_records": len(bundle["ontology"]),
        "training_metric_records": len(bundle["metrics"]),
        "bytes": bundle["bytes"],
    }
