from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ask_ida_cli.catalog import (
    CatalogError,
    canonical_json,
    content_manifest,
    is_safe_relative_file,
    load_catalog_bytes,
)


def catalog_entry(*, status: str = "supported", digest: str | None = "a" * 64) -> dict:
    entry = {
        "artifact_id": "fixture-model",
        "display_name": "Fixture model",
        "kind": "model",
        "status": status,
        "source": {"host": "huggingface.co", "repo_id": "org/model", "revision": "0123456789abcdef"},
    }
    if digest is not None:
        entry["artifact_sha256"] = digest
    return entry


def unsigned_catalog(entry: dict | None = None) -> dict:
    return {
        "schema_version": "ask-ida-cli-dependency-catalog.v1",
        "catalog_version": "test-1",
        "issued_at": "2026-09-05T00:00:00Z",
        "expires_at": "2026-10-05T00:00:00Z",
        "public_hosts": ["huggingface.co"],
        "docs_bundle": {"bundle_id": "docs", "sha256": "b" * 64},
        "models": [entry or catalog_entry()],
    }


def signed(catalog: dict) -> tuple[bytes, str]:
    key = Ed25519PrivateKey.generate()
    payload = canonical_json(catalog)
    envelope = {**catalog, "signature": {"algorithm": "ed25519", "key_id": "test", "value": base64.b64encode(key.sign(payload)).decode("ascii")}}
    public = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("ascii")
    return json.dumps(envelope).encode("utf-8"), public


def test_signed_catalog_is_accepted() -> None:
    payload, public = signed(unsigned_catalog())
    result = load_catalog_bytes(payload, public_key_pem=public)
    assert result["schema_version"] == "ask-ida-cli-dependency-catalog.v1"


def test_unsigned_or_tampered_catalog_is_rejected() -> None:
    payload, public = signed(unsigned_catalog())
    with pytest.raises(CatalogError, match="signature"):
        load_catalog_bytes(payload.replace(b'"value":', b'"value":"tampered", "ignored":'), public_key_pem=public)
    with pytest.raises(CatalogError, match="signature"):
        load_catalog_bytes(json.dumps(unsigned_catalog()).encode(), public_key_pem=public)


def test_private_fields_and_unapproved_host_are_rejected() -> None:
    private = unsigned_catalog()
    private["token"] = "must-not-cross"
    payload, public = signed(private)
    with pytest.raises(CatalogError, match="private"):
        load_catalog_bytes(payload, public_key_pem=public)

    unsafe_host = unsigned_catalog()
    unsafe_host["public_hosts"] = ["example.com"]
    payload, public = signed(unsafe_host)
    with pytest.raises(CatalogError, match="host"):
        load_catalog_bytes(payload, public_key_pem=public)


def test_supported_entry_requires_immutable_digest() -> None:
    payload, public = signed(unsigned_catalog(catalog_entry(digest=None)))
    with pytest.raises(CatalogError, match="artifact_sha256"):
        load_catalog_bytes(payload, public_key_pem=public)


def test_expired_catalog_is_rejected_even_with_a_valid_signature() -> None:
    expired = unsigned_catalog()
    expired["expires_at"] = "2020-01-01T00:00:00Z"
    payload, public = signed(expired)
    with pytest.raises(CatalogError, match="expired"):
        load_catalog_bytes(payload, public_key_pem=public)


def test_safe_file_boundary_rejects_code_and_traversal() -> None:
    assert is_safe_relative_file("config.json")
    assert is_safe_relative_file("weights/model.safetensors")
    assert not is_safe_relative_file("modeling_custom.py")
    assert not is_safe_relative_file("../outside.json")
    assert not is_safe_relative_file("C:/outside.json")


def test_content_manifest_is_content_based(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    first = content_manifest(root)
    (root / "model.safetensors").write_bytes(b"changed")
    second = content_manifest(root)
    assert first["sha256"] != second["sha256"]
    assert first["files"][0]["path"] == second["files"][0]["path"]
