"""Local model discovery and explicit verified downloads."""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .catalog import (
    IMMUTABLE_REVISION_PATTERN,
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_FILE_BYTES,
    MAX_ARTIFACT_FILES,
    CatalogError,
    catalog_entry,
    content_manifest,
)

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MODEL_DIR_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,95}$")
DOWNLOAD_ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.txt",
    "*.jinja",
    "tokenizer*",
    "*.safetensors",
    "*.safetensors.index.json",
)
DOWNLOAD_IGNORE_PATTERNS = (
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


def _snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise CatalogError(
            "model downloads require the runtime extra: pip install 'ida-cli-public[runtime]'",
            code="download_dependency",
        ) from None
    return snapshot_download


def _safe_target(root: Path, artifact_id: str) -> Path:
    if not MODEL_DIR_PATTERN.fullmatch(artifact_id):
        raise CatalogError("artifact_id cannot be used as a local directory", code="artifact_path")
    resolved_root = root.resolve(strict=False)
    target = (resolved_root / artifact_id).resolve(strict=False)
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise CatalogError("artifact target escapes the model root", code="artifact_path") from exc
    return target


def list_local_models(root: Path) -> list[dict[str, Any]]:
    resolved_root = root.resolve(strict=False)
    if not resolved_root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for candidate in sorted(resolved_root.iterdir(), key=lambda item: item.name.casefold()):
        if not candidate.is_dir() or not (candidate / "config.json").is_file():
            continue
        try:
            manifest = content_manifest(candidate)
        except CatalogError as exc:
            rows.append({"directory": candidate.name, "ready": False, "error": exc.code})
            continue
        rows.append({"directory": candidate.name, "ready": True, "artifact_sha256": manifest["sha256"], **manifest})
    return rows


def download_reviewed(catalog: dict[str, Any], artifact_id: str, model_root: Path) -> dict[str, Any]:
    entry = catalog_entry(catalog, artifact_id)
    if entry.get("status") != "supported":
        raise CatalogError(
            "this catalog entry has no immutable artifact attestation yet; use the explicit experimental path if appropriate",
            code="catalog_unattested",
        )
    source = entry["source"]
    if source.get("host") not in catalog.get("public_hosts", []):
        raise CatalogError("catalog source host is not enabled", code="catalog_host")
    if source.get("host") != "huggingface.co":
        raise CatalogError("this release has no resolver for the catalog source host", code="catalog_host")
    if not IMMUTABLE_REVISION_PATTERN.fullmatch(str(source.get("revision") or "")):
        raise CatalogError("reviewed model requires an immutable commit revision", code="catalog_revision")
    target = _safe_target(model_root, str(entry["artifact_id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        manifest = content_manifest(
            target,
            max_file_bytes=int(entry.get("max_file_bytes", MAX_ARTIFACT_FILE_BYTES)),
            max_bytes=int(entry.get("max_bytes", MAX_ARTIFACT_BYTES)),
            max_files=int(entry.get("max_files", MAX_ARTIFACT_FILES)),
        )
        if manifest["sha256"] == entry["artifact_sha256"]:
            return {"artifact_id": artifact_id, "status": "already_verified", "manifest": manifest}
        raise CatalogError("existing model directory does not match the catalog digest", code="artifact_drift")
    stage = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=str(target.parent)))
    try:
        snapshot_download = _snapshot_download()
        snapshot_download(
            repo_id=str(source["repo_id"]),
            revision=str(source["revision"]),
            local_dir=str(stage),
            endpoint="https://huggingface.co",
            token=False,
            # Catalog metadata cannot loosen the release-owned file policy.
            allow_patterns=list(DOWNLOAD_ALLOW_PATTERNS),
            ignore_patterns=list(DOWNLOAD_IGNORE_PATTERNS),
        )
        manifest = content_manifest(
            stage,
            max_file_bytes=int(entry.get("max_file_bytes", MAX_ARTIFACT_FILE_BYTES)),
            max_bytes=int(entry.get("max_bytes", MAX_ARTIFACT_BYTES)),
            max_files=int(entry.get("max_files", MAX_ARTIFACT_FILES)),
        )
        if manifest["sha256"] != entry["artifact_sha256"]:
            raise CatalogError("downloaded artifact does not match the catalog digest", code="artifact_mismatch")
        stage.replace(target)
        return {"artifact_id": artifact_id, "status": "verified", "manifest": manifest}
    except CatalogError:
        raise
    except Exception as exc:  # noqa: BLE001 - upstream downloader errors are normalized at the CLI boundary.
        raise CatalogError(f"reviewed model download failed: {exc}", code="download_failed") from None
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def download_experimental(model_id: str, model_root: Path, *, revision: str | None = None) -> dict[str, Any]:
    if not MODEL_ID_PATTERN.fullmatch(str(model_id or "")):
        raise CatalogError("experimental model must be an upstream namespace/name ID", code="experimental_id")
    target_id = str(model_id).replace("/", "--").casefold()
    target = _safe_target(model_root, target_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise CatalogError("experimental target already exists; remove it explicitly before retrying", code="experimental_existing")
    stage = Path(tempfile.mkdtemp(prefix=f".{target_id}.", dir=str(target.parent)))
    try:
        snapshot_download = _snapshot_download()
        snapshot_download(
            repo_id=model_id,
            revision=revision or "main",
            local_dir=str(stage),
            endpoint="https://huggingface.co",
            token=False,
            allow_patterns=list(DOWNLOAD_ALLOW_PATTERNS),
            ignore_patterns=list(DOWNLOAD_IGNORE_PATTERNS),
        )
        manifest = content_manifest(stage)
        stage.replace(target)
        return {
            "model_id": model_id,
            "status": "experimental_unverified",
            "support": "not_guaranteed",
            "manifest": manifest,
        }
    except CatalogError:
        raise
    except Exception as exc:  # noqa: BLE001 - upstream downloader errors are normalized at the CLI boundary.
        raise CatalogError(f"experimental model download failed: {exc}", code="download_failed") from None
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
