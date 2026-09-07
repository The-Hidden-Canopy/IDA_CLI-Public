from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import ask_ida_cli.catalog as catalog_module
from ask_ida_cli.catalog import (
    CATALOG_KEY_ID,
    CatalogError,
    _HTTPSOnlyRedirectHandler,
    canonical_json,
    content_manifest,
    fetch_catalog,
    is_safe_relative_file,
    load_catalog_bytes,
)

_TEST_PRIVATE_KEY = Ed25519PrivateKey.generate()
_TEST_PUBLIC_KEY = _TEST_PRIVATE_KEY.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("ascii")


@pytest.fixture(autouse=True)
def bundled_test_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog_module, "_bundled_public_key_pem", lambda: _TEST_PUBLIC_KEY)


def catalog_entry(*, status: str = "supported", digest: str | None = "a" * 64) -> dict:
    entry = {
        "artifact_id": "fixture-model",
        "display_name": "Fixture model",
        "kind": "model",
        "status": status,
        "source": {"host": "huggingface.co", "repo_id": "org/model", "revision": "0" * 40},
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
        "cli_releases": [{
            "version": "0.2.0",
            "package_name": "ida-cli-public",
            "python_requires": ">=3.11",
            "docs_bundle_id": "docs",
            "docs_bundle_sha256": "b" * 64,
            "dependency_mode": "catalog_metadata_local_resolution",
        }],
        "docs_bundle": {
            "bundle_id": "docs",
            "sha256": "b" * 64,
            "source": "tests/test_catalog.py",
            "status": "bundled_and_cataloged",
        },
        "models": [entry or catalog_entry()],
    }


def signed(catalog: dict) -> tuple[bytes, str]:
    payload = canonical_json(catalog)
    envelope = {**catalog, "signature": {"algorithm": "ed25519", "key_id": CATALOG_KEY_ID, "value": base64.b64encode(_TEST_PRIVATE_KEY.sign(payload)).decode("ascii")}}
    return json.dumps(envelope).encode("utf-8"), _TEST_PUBLIC_KEY


def test_signed_catalog_is_accepted() -> None:
    payload, _ = signed(unsigned_catalog())
    result = load_catalog_bytes(payload)
    assert result["schema_version"] == "ask-ida-cli-dependency-catalog.v1"


def test_unsigned_or_tampered_catalog_is_rejected() -> None:
    payload, _ = signed(unsigned_catalog())
    with pytest.raises(CatalogError, match="signature"):
        load_catalog_bytes(payload.replace(b'"value":', b'"value":"tampered", "ignored":'))
    with pytest.raises(CatalogError, match="signature"):
        load_catalog_bytes(json.dumps(unsigned_catalog()).encode())


def test_private_fields_and_unapproved_host_are_rejected() -> None:
    private = unsigned_catalog()
    private["token"] = "must-not-cross"
    payload, _ = signed(private)
    with pytest.raises(CatalogError, match="private"):
        load_catalog_bytes(payload)

    unsafe_host = unsigned_catalog()
    unsafe_host["public_hosts"] = ["example.com"]
    payload, _ = signed(unsafe_host)
    with pytest.raises(CatalogError, match="host"):
        load_catalog_bytes(payload)


def test_supported_entry_requires_immutable_digest() -> None:
    payload, _ = signed(unsigned_catalog(catalog_entry(digest=None)))
    with pytest.raises(CatalogError, match="artifact_sha256"):
        load_catalog_bytes(payload)

    mutable = catalog_entry()
    mutable["source"]["revision"] = "main"
    payload, _ = signed(unsigned_catalog(mutable))
    with pytest.raises(CatalogError, match="immutable commit"):
        load_catalog_bytes(payload)


def test_expired_catalog_is_rejected_even_with_a_valid_signature() -> None:
    expired = unsigned_catalog()
    expired["expires_at"] = "2020-01-01T00:00:00Z"
    payload, _ = signed(expired)
    with pytest.raises(CatalogError, match="expired"):
        load_catalog_bytes(payload)


def test_future_or_inverted_catalog_timestamps_are_rejected() -> None:
    future = unsigned_catalog()
    future["issued_at"] = "2099-01-01T00:00:00Z"
    payload, _ = signed(future)
    with pytest.raises(CatalogError, match="issued_at"):
        load_catalog_bytes(payload)

    inverted = unsigned_catalog()
    inverted["expires_at"] = "2026-09-04T00:00:00Z"
    payload, _ = signed(inverted)
    with pytest.raises(CatalogError, match="expired"):
        load_catalog_bytes(payload)


def test_catalog_key_override_is_not_taken_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    payload, _ = signed(unsigned_catalog())
    monkeypatch.setenv("ASK_IDA_PUBLIC_CATALOG_PUBLIC_KEY", "not-used")
    assert load_catalog_bytes(payload)["schema_version"] == "ask-ida-cli-dependency-catalog.v1"


def test_catalog_verification_key_is_not_a_runtime_argument() -> None:
    payload, public = signed(unsigned_catalog())
    with pytest.raises(TypeError, match="public_key_pem"):
        load_catalog_bytes(payload, public_key_pem=public)


def test_missing_release_pinned_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    payload, _ = signed(unsigned_catalog())
    monkeypatch.setattr(catalog_module, "_bundled_public_key_pem", lambda: None)
    with pytest.raises(CatalogError, match="verification key"):
        load_catalog_bytes(payload)


def test_private_aliases_and_unknown_fields_are_rejected_at_every_catalog_level() -> None:
    probes = [
        ("root private alias", lambda value: value.__setitem__("privateKey", "sentinel")),
        ("release private alias", lambda value: value["cli_releases"][0].__setitem__("secret_key", "sentinel")),
        ("docs private alias", lambda value: value["docs_bundle"].__setitem__("apiKey", "sentinel")),
        ("model private alias", lambda value: value["models"][0].__setitem__("bearer_token", "sentinel")),
        ("source private alias", lambda value: value["models"][0]["source"].__setitem__("authorization", "sentinel")),
        ("signature unknown field", lambda value: value.__setitem__(
            "signature", {"algorithm": "ed25519", "key_id": CATALOG_KEY_ID, "value": "sentinel", "extra": "sentinel"}
        )),
        ("top-level unknown field", lambda value: value.__setitem__("unexpected", "sentinel")),
    ]
    for label, mutate in probes:
        candidate = json.loads(json.dumps(unsigned_catalog()))
        mutate(candidate)
        payload, _ = signed(candidate)
        if label == "signature unknown field":
            envelope = json.loads(payload)
            envelope["signature"]["extra"] = "sentinel"
            payload = json.dumps(envelope).encode()
        with pytest.raises(CatalogError, match="private|unknown"):
            load_catalog_bytes(payload)


def test_catalog_shape_matches_hub_required_fields_and_types() -> None:
    mutations = [
        ("missing models", lambda value: value.pop("models"), "catalog_entries"),
        ("missing releases", lambda value: value.pop("cli_releases"), "catalog_releases"),
        ("duplicate host", lambda value: value.__setitem__("public_hosts", ["huggingface.co", "huggingface.co"]), "catalog_hosts"),
        ("string limit", lambda value: value["models"][0].__setitem__("max_files", "1"), "catalog_limits"),
        ("invalid optional digest", lambda value: value["models"][0].__setitem__("artifact_sha256", "not-a-digest"), "catalog_attestation"),
    ]
    for label, mutate, code in mutations:
        candidate = unsigned_catalog()
        mutate(candidate)
        payload, _ = signed(candidate)
        with pytest.raises(CatalogError) as failure:
            load_catalog_bytes(payload)
        assert failure.value.code == code, label


def test_catalog_canonicalization_uses_utf8_for_public_unicode() -> None:
    assert canonical_json({"display_name": "Café", "nested": {"é": "naïve"}}).hex() == (
        "7b22646973706c61795f6e616d65223a22436166c3a9222c226e6573746564223a7b22c3a9223a226e61c3af7665227d7d"
    )


def test_catalog_transport_rejects_http_and_http_redirects() -> None:
    with pytest.raises(CatalogError, match="HTTPS"):
        fetch_catalog("http://example.com/catalog")

    request = urllib.request.Request("https://example.com/catalog")
    with pytest.raises(CatalogError, match="HTTPS"):
        _HTTPSOnlyRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "http://example.com/catalog"
        )


def test_safe_file_boundary_rejects_code_and_traversal() -> None:
    assert is_safe_relative_file("config.json")
    assert is_safe_relative_file("weights/model.safetensors")
    assert not is_safe_relative_file("modeling_custom.py")
    assert not is_safe_relative_file("pytorch_model.bin")
    assert not is_safe_relative_file("pytorch_model.bin.index.json")
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
