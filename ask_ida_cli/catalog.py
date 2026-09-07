"""Signed public dependency catalog and deterministic local attestations."""

from __future__ import annotations

import base64
import binascii
import fnmatch
import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .errors import CLIError

SCHEMA_VERSION = "ask-ida-cli-dependency-catalog.v1"
CATALOG_KEY_ID = "ida-cli-public-2026-09"
MAX_CATALOG_BYTES = 2 * 1024 * 1024
MAX_CATALOG_ENTRIES = 500
MAX_ARTIFACT_FILE_BYTES = 64 * 1024**3
MAX_ARTIFACT_BYTES = 512 * 1024**3
MAX_ARTIFACT_FILES = 100_000
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
IMMUTABLE_REVISION_PATTERN = re.compile(r"^[a-f0-9]{40}$")
SAFE_HOSTS = frozenset({"huggingface.co"})
CATALOG_KEYS = frozenset(
    {
        "schema_version",
        "catalog_version",
        "issued_at",
        "expires_at",
        "public_hosts",
        "cli_releases",
        "docs_bundle",
        "models",
        "signature",
    }
)
RELEASE_KEYS = frozenset(
    {
        "version",
        "package_name",
        "python_requires",
        "docs_bundle_id",
        "docs_bundle_sha256",
        "dependency_mode",
    }
)
DOCS_BUNDLE_KEYS = frozenset({"bundle_id", "sha256", "source", "status"})
MODEL_KEYS = frozenset(
    {
        "artifact_id",
        "display_name",
        "kind",
        "status",
        "support",
        "source",
        "artifact_sha256",
        "max_file_bytes",
        "max_bytes",
        "max_files",
    }
)
SOURCE_KEYS = frozenset({"host", "repo_id", "revision"})
SIGNATURE_KEYS = frozenset({"algorithm", "key_id", "value"})
PRIVATE_KEY_NAMES = frozenset(
    {
        "orgid",
        "organizationid",
        "workertoken",
        "token",
        "secret",
        "privatekey",
        "secretkey",
        "apikey",
        "localpath",
        "checkpointpath",
        "prompt",
        "rawlog",
        "rawlogs",
        "password",
        "accesstoken",
        "credential",
        "clientsecret",
        "bearertoken",
        "authtoken",
        "authorization",
        "cookie",
        "privateruntime",
        "workersubject",
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
)
DENIED_FILE_PATTERNS = (
    "*.py",
    "*.pyc",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.gguf",
    "*.bin",
    "*.bin.index.json",
)


class CatalogError(CLIError):
    """A catalog, source, or artifact verification failure."""

    def __init__(self, message: str, *, code: str = "catalog_invalid", exit_code: int = 3) -> None:
        super().__init__(message, code=code, exit_code=exit_code)


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject catalog redirects that leave HTTPS transport."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target = urlsplit(newurl)
        if target.scheme.casefold() != "https" or not target.netloc:
            raise CatalogError("catalog redirect must use HTTPS", code="catalog_transport")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _without_signature(envelope: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(envelope)
    unsigned.pop("signature", None)
    return unsigned


def _bundled_public_key_pem() -> str | None:
    try:
        return files("ask_ida_cli").joinpath("catalog_public_key.pem").read_text(encoding="ascii")
    except (FileNotFoundError, OSError):
        return None


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _contains_private(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_private(item) for item in value)
    if not isinstance(value, dict):
        return False
    for key, child in value.items():
        normalized = _normalize_key(key)
        if normalized in PRIVATE_KEY_NAMES or any(part in normalized for part in ("password", "accesstoken", "credential")):
            return True
        if _contains_private(child):
            return True
    return False


def _assert_exact_keys(value: Any, allowed: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{label} must be an object", code="catalog_schema")
    if any(key not in allowed for key in value):
        raise CatalogError(f"{label} contains an unknown field", code="catalog_schema")
    return value


def validate_catalog_shape(envelope: dict[str, Any]) -> None:
    if not isinstance(envelope, dict) or envelope.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError(f"unsupported catalog schema; expected {SCHEMA_VERSION}", code="catalog_schema")
    if _contains_private(envelope):
        raise CatalogError("catalog contains private fields", code="catalog_private_field")
    _assert_exact_keys(envelope, CATALOG_KEYS, "catalog")
    if "signature" in envelope:
        signature = _assert_exact_keys(envelope["signature"], SIGNATURE_KEYS, "catalog signature")
        if (
            signature.get("algorithm") != "ed25519"
            or not isinstance(signature.get("key_id"), str)
            or not isinstance(signature.get("value"), str)
        ):
            raise CatalogError("catalog signature is invalid", code="catalog_signature")
    if any(not isinstance(envelope.get(key), str) for key in ("catalog_version", "issued_at", "expires_at")):
        raise CatalogError("catalog metadata is invalid", code="catalog_schema")
    expires_at = envelope["expires_at"]
    now = datetime.now(UTC)
    try:
        issued = datetime.fromisoformat(envelope["issued_at"])
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        raise CatalogError("catalog issued_at or expires_at is invalid", code="catalog_expiry") from None
    if issued.tzinfo is None or expiry.tzinfo is None or issued > now + timedelta(minutes=5):
        raise CatalogError("catalog issued_at is invalid", code="catalog_expiry")
    if expiry <= now or expiry <= issued:
        raise CatalogError("catalog is expired", code="catalog_expired")
    docs_bundle = _assert_exact_keys(envelope.get("docs_bundle"), DOCS_BUNDLE_KEYS, "catalog docs bundle")
    if any(not isinstance(docs_bundle.get(key), str) for key in DOCS_BUNDLE_KEYS):
        raise CatalogError("catalog docs_bundle is invalid", code="catalog_docs")
    if not SHA256_PATTERN.fullmatch(docs_bundle["sha256"]):
        raise CatalogError("catalog docs_bundle attestation is missing", code="catalog_docs")
    releases = envelope.get("cli_releases")
    if not isinstance(releases, list) or not releases:
        raise CatalogError("catalog releases are invalid", code="catalog_releases")
    for release_value in releases:
        release = _assert_exact_keys(release_value, RELEASE_KEYS, "catalog release")
        if any(not isinstance(release.get(key), str) for key in RELEASE_KEYS):
            raise CatalogError("catalog release is invalid", code="catalog_releases")
        if (
            not SHA256_PATTERN.fullmatch(release["docs_bundle_sha256"])
            or release["docs_bundle_sha256"] != docs_bundle["sha256"]
        ):
            raise CatalogError("catalog release documentation attestation is invalid", code="catalog_releases")
    entries = envelope.get("models")
    if not isinstance(entries, list) or len(entries) > MAX_CATALOG_ENTRIES:
        raise CatalogError("catalog model entries are invalid or too numerous", code="catalog_entries")
    hosts = envelope.get("public_hosts")
    if hosts != ["huggingface.co"]:
        raise CatalogError("catalog public host policy is invalid", code="catalog_hosts")
    seen: set[str] = set()
    for entry_value in entries:
        entry = _assert_exact_keys(entry_value, MODEL_KEYS, "catalog model entry")
        artifact_id = entry.get("artifact_id") if isinstance(entry.get("artifact_id"), str) else ""
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,95}", artifact_id):
            raise CatalogError("catalog artifact_id is invalid", code="catalog_entry")
        if artifact_id in seen:
            raise CatalogError("catalog artifact_id is duplicated", code="catalog_entry")
        seen.add(artifact_id)
        if entry.get("kind") != "model":
            raise CatalogError("catalog entry kind is invalid", code="catalog_entry")
        if any(
            key in entry and not isinstance(entry[key], str)
            for key in ("display_name", "support")
        ):
            raise CatalogError("catalog model entry is invalid", code="catalog_entry")
        source = _assert_exact_keys(entry.get("source"), SOURCE_KEYS, "catalog model source")
        if any(not isinstance(source.get(key), str) for key in SOURCE_KEYS) or source.get("host") != hosts[0]:
            raise CatalogError("catalog source host is not allowed", code="catalog_source")
        repo_id = source["repo_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
            raise CatalogError("catalog source repo_id is invalid", code="catalog_source")
        revision = source["revision"]
        if not revision or len(revision) > 200 or any(char in revision for char in ("/", "\\", "..")):
            raise CatalogError("catalog source revision is invalid", code="catalog_source")
        status = str(entry.get("status") or "")
        if "artifact_sha256" in entry:
            artifact_digest = entry["artifact_sha256"]
            if not isinstance(artifact_digest, str) or not SHA256_PATTERN.fullmatch(artifact_digest):
                raise CatalogError("catalog artifact_sha256 is invalid", code="catalog_attestation")
        if status == "supported":
            digest = entry.get("artifact_sha256") if isinstance(entry.get("artifact_sha256"), str) else ""
            if not SHA256_PATTERN.fullmatch(digest):
                raise CatalogError("supported catalog entries require an artifact_sha256", code="catalog_attestation")
            if not IMMUTABLE_REVISION_PATTERN.fullmatch(revision):
                raise CatalogError("supported catalog entries require an immutable commit revision", code="catalog_revision")
        elif status not in {"catalog_only", "deprecated"}:
            raise CatalogError("catalog entry status is invalid", code="catalog_status")
        for limit_name, maximum in (
            ("max_file_bytes", MAX_ARTIFACT_FILE_BYTES),
            ("max_bytes", MAX_ARTIFACT_BYTES),
            ("max_files", MAX_ARTIFACT_FILES),
        ):
            if limit_name in entry:
                limit = entry[limit_name]
                if isinstance(limit, bool) or not isinstance(limit, int):
                    raise CatalogError(f"catalog {limit_name} is invalid", code="catalog_limits") from None
                if limit <= 0 or limit > maximum:
                    raise CatalogError(f"catalog {limit_name} exceeds release bounds", code="catalog_limits")


def verify_catalog_signature(envelope: dict[str, Any]) -> None:
    validate_catalog_shape(envelope)
    signature = envelope.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise CatalogError("catalog signature is missing or unsupported", code="catalog_signature")
    if signature.get("key_id") != CATALOG_KEY_ID:
        raise CatalogError("catalog signature key identity is not pinned to this release", code="catalog_key_id")
    encoded = signature.get("value")
    if not isinstance(encoded, str):
        raise CatalogError("catalog signature value is missing", code="catalog_signature")
    key_material = _bundled_public_key_pem()
    if not key_material:
        raise CatalogError("catalog verification key is not configured", code="catalog_key")
    try:
        key = serialization.load_pem_public_key(
            key_material.encode("utf-8") if isinstance(key_material, str) else key_material
        )
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("verification key is not Ed25519")
        key.verify(base64.b64decode(encoded, validate=True), canonical_json(_without_signature(envelope)))
    except (binascii.Error, InvalidSignature, TypeError, ValueError) as exc:
        raise CatalogError(f"catalog signature verification failed: {exc}", code="catalog_signature") from None


def load_catalog_bytes(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_CATALOG_BYTES:
        raise CatalogError("catalog response is too large", code="catalog_size")
    try:
        envelope = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"catalog is not valid JSON: {exc}", code="catalog_json") from None
    if not isinstance(envelope, dict):
        raise CatalogError("catalog response must be a JSON object", code="catalog_json")
    verify_catalog_signature(envelope)
    return envelope


def fetch_catalog(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    target = urlsplit(url)
    if target.scheme.casefold() != "https" or not target.netloc:
        raise CatalogError("catalog URL must use HTTPS", code="catalog_transport")
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "IDA_CLI-Public/0.2"})
    opener = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(MAX_CATALOG_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CatalogError(f"catalog request failed: {exc}", code="catalog_unavailable") from None
    return load_catalog_bytes(payload)


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


def content_manifest(
    root: Path,
    *,
    max_file_bytes: int = MAX_ARTIFACT_FILE_BYTES,
    max_bytes: int = MAX_ARTIFACT_BYTES,
    max_files: int = MAX_ARTIFACT_FILES,
) -> dict[str, Any]:
    for name, limit, maximum in (
        ("max_file_bytes", max_file_bytes, MAX_ARTIFACT_FILE_BYTES),
        ("max_bytes", max_bytes, MAX_ARTIFACT_BYTES),
        ("max_files", max_files, MAX_ARTIFACT_FILES),
    ):
        if not isinstance(limit, int) or limit <= 0 or limit > maximum:
            raise CatalogError(f"{name} exceeds release bounds", code="artifact_limits")
    resolved_root = root.resolve(strict=True)
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in resolved_root.rglob("*") if item.is_file()):
        if len(files) >= max_files:
            raise CatalogError("artifact contains too many files", code="artifact_limits")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise CatalogError("artifact contains a path outside its root", code="artifact_path") from exc
        if not is_safe_relative_file(relative):
            raise CatalogError(f"artifact contains a disallowed file: {relative}", code="artifact_file")
        size = path.stat().st_size
        if size > max_file_bytes or total_bytes + size > max_bytes:
            raise CatalogError("artifact exceeds release size limits", code="artifact_limits")
        digest = hashlib.sha256()
        actual_size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                actual_size += len(chunk)
                digest.update(chunk)
        if actual_size > max_file_bytes or total_bytes + actual_size > max_bytes:
            raise CatalogError("artifact exceeds release size limits", code="artifact_limits")
        total_bytes += actual_size
        files.append({"path": relative, "bytes": actual_size, "sha256": digest.hexdigest()})
    if not files:
        raise CatalogError("artifact contains no allowed files", code="artifact_empty")
    manifest = {"files": files, "file_count": len(files), "bytes": total_bytes}
    manifest["sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest
