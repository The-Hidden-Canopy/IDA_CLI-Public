"""Signed public dependency catalog and deterministic local attestations."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .errors import CLIError

SCHEMA_VERSION = "ask-ida-cli-dependency-catalog.v1"
MAX_CATALOG_BYTES = 2 * 1024 * 1024
MAX_CATALOG_ENTRIES = 500
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
SAFE_HOSTS = frozenset({"huggingface.co", "hf.co"})
PRIVATE_KEY_NAMES = frozenset(
    {
        "org_id",
        "organization_id",
        "worker_token",
        "token",
        "secret",
        "private_key",
        "api_key",
        "local_path",
        "checkpoint_path",
        "prompt",
        "raw_log",
        "raw_logs",
    }
)
SAFE_FILE_PATTERNS = (
    "*.json",
    "*.model",
    "*.txt",
    "*.jinja",
    "tokenizer*",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
)
DENIED_FILE_PATTERNS = ("*.py", "*.pyc", "*.pyd", "*.so", "*.dll", "*.h5", "*.msgpack", "*.onnx", "*.gguf")


class CatalogError(CLIError):
    """A catalog, source, or artifact verification failure."""

    def __init__(self, message: str, *, code: str = "catalog_invalid", exit_code: int = 3) -> None:
        super().__init__(message, code=code, exit_code=exit_code)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _without_signature(envelope: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(envelope)
    unsigned.pop("signature", None)
    return unsigned


def _contains_private(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_private(item) for item in value)
    if not isinstance(value, dict):
        return False
    for key, child in value.items():
        normalized = str(key).casefold()
        if normalized in PRIVATE_KEY_NAMES or any(part in normalized for part in ("password", "access_token", "credential")):
            return True
        if _contains_private(child):
            return True
    return False


def validate_catalog_shape(envelope: dict[str, Any]) -> None:
    if not isinstance(envelope, dict) or envelope.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError(f"unsupported catalog schema; expected {SCHEMA_VERSION}", code="catalog_schema")
    if _contains_private(envelope):
        raise CatalogError("catalog contains private fields", code="catalog_private_field")
    expires_at = str(envelope.get("expires_at") or "")
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        raise CatalogError("catalog expires_at is invalid", code="catalog_expiry") from None
    if expiry.tzinfo is None or expiry <= datetime.now(UTC):
        raise CatalogError("catalog is expired", code="catalog_expired")
    docs_bundle = envelope.get("docs_bundle")
    if not isinstance(docs_bundle, dict) or not SHA256_PATTERN.fullmatch(str(docs_bundle.get("sha256") or "")):
        raise CatalogError("catalog docs_bundle attestation is missing", code="catalog_docs")
    entries = envelope.get("models", [])
    if not isinstance(entries, list) or len(entries) > MAX_CATALOG_ENTRIES:
        raise CatalogError("catalog model entries are invalid or too numerous", code="catalog_entries")
    hosts = envelope.get("public_hosts")
    if not isinstance(hosts, list) or not hosts or any(not isinstance(host, str) for host in hosts):
        raise CatalogError("catalog public_hosts is required", code="catalog_hosts")
    for host in hosts:
        if host.casefold() not in SAFE_HOSTS:
            raise CatalogError(f"catalog host is not enabled by this release: {host}", code="catalog_host")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise CatalogError("catalog model entry must be an object", code="catalog_entry")
        artifact_id = str(entry.get("artifact_id") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,95}", artifact_id):
            raise CatalogError("catalog artifact_id is invalid", code="catalog_entry")
        if artifact_id in seen:
            raise CatalogError("catalog artifact_id is duplicated", code="catalog_entry")
        seen.add(artifact_id)
        if str(entry.get("kind") or "") != "model":
            raise CatalogError("catalog entry kind is invalid", code="catalog_entry")
        source = entry.get("source")
        if not isinstance(source, dict) or source.get("host") not in hosts:
            raise CatalogError("catalog source host is not allowed", code="catalog_source")
        repo_id = str(source.get("repo_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
            raise CatalogError("catalog source repo_id is invalid", code="catalog_source")
        revision = str(source.get("revision") or "")
        if not revision or len(revision) > 200 or any(char in revision for char in ("/", "\\", "..")):
            raise CatalogError("catalog source revision is invalid", code="catalog_source")
        status = str(entry.get("status") or "")
        if status == "supported":
            digest = str(entry.get("artifact_sha256") or "")
            if not SHA256_PATTERN.fullmatch(digest):
                raise CatalogError("supported catalog entries require an artifact_sha256", code="catalog_attestation")
        elif status not in {"catalog_only", "deprecated"}:
            raise CatalogError("catalog entry status is invalid", code="catalog_status")


def verify_catalog_signature(envelope: dict[str, Any], public_key_pem: str | bytes | None) -> None:
    validate_catalog_shape(envelope)
    signature = envelope.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise CatalogError("catalog signature is missing or unsupported", code="catalog_signature")
    encoded = signature.get("value")
    if not isinstance(encoded, str):
        raise CatalogError("catalog signature value is missing", code="catalog_signature")
    key_material = public_key_pem or os.environ.get("ASK_IDA_PUBLIC_CATALOG_PUBLIC_KEY")
    if not key_material:
        raise CatalogError("catalog verification key is not configured", code="catalog_key")
    try:
        key = serialization.load_pem_public_key(
            key_material.encode("utf-8") if isinstance(key_material, str) else key_material
        )
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("verification key is not Ed25519")
        key.verify(base64.b64decode(encoded, validate=True), canonical_json(_without_signature(envelope)))
    except Exception as exc:  # cryptography intentionally normalizes malformed signatures here.
        raise CatalogError(f"catalog signature verification failed: {exc}", code="catalog_signature") from None


def load_catalog_bytes(payload: bytes, *, public_key_pem: str | bytes | None = None) -> dict[str, Any]:
    if len(payload) > MAX_CATALOG_BYTES:
        raise CatalogError("catalog response is too large", code="catalog_size")
    try:
        envelope = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"catalog is not valid JSON: {exc}", code="catalog_json") from None
    if not isinstance(envelope, dict):
        raise CatalogError("catalog response must be a JSON object", code="catalog_json")
    verify_catalog_signature(envelope, public_key_pem)
    return envelope


def fetch_catalog(url: str, *, public_key_pem: str | bytes | None = None, timeout: float = 10.0) -> dict[str, Any]:
    if not url.startswith("https://"):
        raise CatalogError("catalog URL must use HTTPS", code="catalog_transport")
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "IDA_CLI-Public/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(MAX_CATALOG_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CatalogError(f"catalog request failed: {exc}", code="catalog_unavailable") from None
    return load_catalog_bytes(payload, public_key_pem=public_key_pem)


def catalog_entry(catalog: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    needle = str(artifact_id or "").strip().casefold()
    for entry in catalog.get("models", []):
        if str(entry.get("artifact_id") or "").casefold() == needle:
            return entry
    raise CatalogError(f"no catalog entry named {artifact_id!r}", code="catalog_entry_missing")


def is_safe_relative_file(relative: str) -> bool:
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", relative) or ".." in path.parts or not relative or "\x00" in relative:
        return False
    name = path.as_posix()
    if any(fnmatch.fnmatch(name, pattern) for pattern in DENIED_FILE_PATTERNS):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in SAFE_FILE_PATTERNS)


def content_manifest(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in resolved_root.rglob("*") if item.is_file()):
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise CatalogError("artifact contains a path outside its root", code="artifact_path") from exc
        if not is_safe_relative_file(relative):
            raise CatalogError(f"artifact contains a disallowed file: {relative}", code="artifact_file")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        total_bytes += len(data)
        files.append({"path": relative, "bytes": len(data), "sha256": digest})
    if not files:
        raise CatalogError("artifact contains no allowed files", code="artifact_empty")
    manifest = {"files": files, "file_count": len(files), "bytes": total_bytes}
    manifest["sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest
