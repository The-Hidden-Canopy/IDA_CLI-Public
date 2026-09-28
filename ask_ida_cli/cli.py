"""Command-line entrypoints for the public Ask IDA surface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import CatalogError, fetch_catalog
from .config import Settings, load_settings
from .docs import explain, verify_bundled_docs
from .errors import CLIError
from .inference import ask_local
from .leaderboard import fetch_public_leaderboard
from .models import download_experimental, download_reviewed, list_local_models
from .telemetry import load_share_bundle, share_kernel_telemetry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ida", description="Public local-first Ask IDA CLI.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--root", help="local CLI root")
    parser.add_argument("--model-root", help="local model root")
    parser.add_argument("--catalog-url", help="Hub catalog URL; network use is explicit per command")
    parser.add_argument("--leaderboard-url", help="public leaderboard URL; network use is explicit per command")
    parser.add_argument("--telemetry-url", help="HTTPS endpoint; used only by an explicit kernel telemetry share command")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    def add_json_flag(child: argparse.ArgumentParser) -> None:
        child.add_argument("--json", dest="json", action="store_true", default=argparse.SUPPRESS, help="emit JSON")

    status = commands.add_parser("status", help="show local public CLI state")
    add_json_flag(status)
    catalog = commands.add_parser("catalog", help="fetch and verify the public Hub catalog")
    add_json_flag(catalog)
    leaderboard = commands.add_parser("leaderboard", help="read the public Neural Forge leaderboard")
    leaderboard_commands = leaderboard.add_subparsers(dest="leaderboard_command", required=True)
    leaderboard_list = leaderboard_commands.add_parser(
        "list", help="fetch confirmed public results; this command has no write or submission path"
    )
    leaderboard_list.add_argument("--url", help="HTTPS public leaderboard URL; otherwise --leaderboard-url or ASK_IDA_PUBLIC_LEADERBOARD_URL")
    leaderboard_list.add_argument("--view", choices=("all", "throughput", "outcome"), default="all")
    leaderboard_list.add_argument("--model-id")
    leaderboard_list.add_argument("--revision")
    leaderboard_list.add_argument("--gpu-vendor", choices=("nvidia", "amd"))
    leaderboard_list.add_argument("--gpu-model")
    leaderboard_list.add_argument("--search")
    leaderboard_list.add_argument(
        "--search-field",
        choices=("all", "model_id", "revision", "gpu_vendor", "gpu_model", "user_id", "metric_id"),
        default="all",
    )
    leaderboard_list.add_argument("--limit", type=int, default=20)
    leaderboard_list.add_argument("--offset", type=int, default=0)
    add_json_flag(leaderboard_list)
    explain_parser = commands.add_parser("explain", help="search bundled public documentation")
    explain_parser.add_argument("query", nargs="+")
    add_json_flag(explain_parser)
    model = commands.add_parser("model", help="inspect or explicitly download local models")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_list = model_commands.add_parser("list", help="list complete local model directories")
    add_json_flag(model_list)
    download = model_commands.add_parser("download", help="download one reviewed or experimental model")
    download.add_argument("model_id", nargs="?")
    download.add_argument("--experimental", action="store_true")
    download.add_argument("--revision")
    add_json_flag(download)
    ask = commands.add_parser("ask", help="run local-only inference")
    ask.add_argument("model_path")
    ask.add_argument("prompt", nargs="+")
    ask.add_argument("--max-new-tokens", type=int, default=256)
    add_json_flag(ask)
    telemetry = commands.add_parser("telemetry", help="inspect or explicitly share Foundry kernel telemetry")
    telemetry_commands = telemetry.add_subparsers(dest="telemetry_command", required=True)
    telemetry_status = telemetry_commands.add_parser("status", help="show the local telemetry sharing boundary")
    add_json_flag(telemetry_status)
    telemetry_share = telemetry_commands.add_parser("share-kernel", help="explicitly POST a validated Foundry kernel telemetry directory")
    telemetry_share.add_argument("input", help="Foundry kernel telemetry directory containing receipt.json")
    telemetry_share.add_argument("--url", help="HTTPS endpoint; otherwise --telemetry-url or ASK_IDA_PUBLIC_TELEMETRY_URL")
    telemetry_share.add_argument(
        "--send-token",
        action="store_true",
        help="send ASK_IDA_PUBLIC_TELEMETRY_TOKEN; requires an explicit --url",
    )
    telemetry_share.add_argument("--dry-run", action="store_true", help="validate and print the bundle without network access")
    add_json_flag(telemetry_share)
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    try:
        return load_settings(
            root=args.root,
            model_root=args.model_root,
            catalog_url=args.catalog_url,
            leaderboard_url=args.leaderboard_url,
            telemetry_url=args.telemetry_url,
        )
    except (ValueError, OSError) as exc:
        raise CLIError(str(exc), code="config_invalid") from None


def _emit(value: Any, *, as_json: bool, human: str | None = None) -> None:
    if as_json or human is None:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(human)


def _run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if args.command == "status":
        value = {
            "version": __version__,
            "root": str(settings.root),
            "model_root": str(settings.model_root),
            "catalog_url": settings.catalog_url,
            "leaderboard_url": settings.leaderboard_url,
            "network": "explicit_catalog_or_download_only",
            "leaderboard_network": "explicit_public_read_only",
            "private_runtime": "not_available_in_public_cli",
            "models": list_local_models(settings.model_root),
        }
        _emit(value, as_json=args.json, human=f"IDA_CLI-Public {__version__}\nModels: {len(value['models'])}")
        return 0
    if args.command == "telemetry":
        if args.telemetry_command == "status":
            value = {
                "sharing": "disabled_by_default",
                "network": "explicit_kernel_telemetry_share_only",
                "schema_version": "native-kernel-telemetry.v2",
                "scope": "kernel_ontology_and_training_metrics",
                "endpoint_configured": bool(settings.telemetry_url),
                "credentials": "not sent unless --send-token is paired with an explicit --url",
                "private_data": "never accepted by the share schema",
            }
            _emit(value, as_json=args.json, human="Telemetry sharing: disabled by default")
            return 0
        if args.telemetry_command == "share-kernel":
            if args.dry_run:
                value = {"status": "validated", "bundle": load_share_bundle(args.input)}
            else:
                if args.send_token and not args.url:
                    raise CLIError(
                        "--send-token requires an explicit --url",
                        code="telemetry_token_requires_url",
                    )
                endpoint = args.url or settings.telemetry_url
                if not endpoint:
                    raise CLIError(
                        "telemetry share-kernel requires --url, --telemetry-url, or ASK_IDA_PUBLIC_TELEMETRY_URL",
                        code="telemetry_url_required",
                    )
                value = share_kernel_telemetry(
                    args.input,
                    endpoint,
                    timeout=settings.timeout_seconds,
                    send_token=args.send_token,
                )
            _emit(value, as_json=args.json, human=str(value.get("status")))
            return 0
    if args.command == "leaderboard":
        if args.leaderboard_command == "list":
            value = fetch_public_leaderboard(
                args.url or settings.leaderboard_url,
                timeout=settings.timeout_seconds,
                view=args.view,
                model_id=args.model_id,
                revision=args.revision,
                gpu_vendor=args.gpu_vendor,
                gpu_model=args.gpu_model,
                search=args.search,
                search_field=args.search_field,
                limit=args.limit,
                offset=args.offset,
            )
            _emit(value, as_json=args.json, human=json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    if args.command == "catalog":
        catalog = fetch_catalog(
            settings.catalog_url,
            timeout=settings.timeout_seconds,
        )
        try:
            verify_bundled_docs(catalog)
        except ValueError as exc:
            raise CatalogError(str(exc), code="catalog_docs") from None
        value = {
            "schema_version": catalog["schema_version"],
            "catalog_version": catalog.get("catalog_version"),
            "expires_at": catalog.get("expires_at"),
            "models": [
                {
                    "artifact_id": item.get("artifact_id"),
                    "status": item.get("status"),
                    "display_name": item.get("display_name"),
                }
                for item in catalog.get("models", [])
            ],
        }
        _emit(value, as_json=args.json, human=json.dumps(value, indent=2))
        return 0
    if args.command == "explain":
        rows = explain(" ".join(args.query))
        value = {"source_state": "bundled_public_docs", "results": rows}
        _emit(value, as_json=args.json, human="\n\n".join(f"{row['title']}\n{row['body']}" for row in rows) or "No public documentation match.")
        return 0
    if args.command == "model":
        if args.model_command == "list":
            rows = list_local_models(settings.model_root)
            _emit({"models": rows}, as_json=args.json, human="\n".join(row["directory"] for row in rows) or "(no complete local models)")
            return 0
        if args.model_command == "download":
            if not args.model_id:
                raise CLIError("model download requires an ID", code="model_id_required")
            if args.experimental:
                result = download_experimental(args.model_id, settings.model_root, revision=args.revision)
            else:
                catalog = fetch_catalog(
                    settings.catalog_url,
                    timeout=settings.timeout_seconds,
                )
                try:
                    verify_bundled_docs(catalog)
                except ValueError as exc:
                    raise CatalogError(str(exc), code="catalog_docs") from None
                result = download_reviewed(catalog, args.model_id, settings.model_root)
            _emit(result, as_json=args.json, human=f"{result.get('status')}: {result.get('artifact_id') or result.get('model_id')}")
            return 0
    if args.command == "ask":
        model_path = Path(args.model_path).expanduser().resolve(strict=False)
        model_root = settings.model_root.resolve(strict=False)
        try:
            model_path.relative_to(model_root)
        except ValueError:
            raise CLIError("model path must remain inside the configured public model root", code="model_path") from None
        result = ask_local(model_path, " ".join(args.prompt), max_new_tokens=args.max_new_tokens)
        _emit({"response": result, "source_state": "local_model"}, as_json=args.json, human=result)
        return 0
    raise CLIError("unsupported command", code="command_invalid")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except CLIError as exc:
        payload = {"error": exc.message, "code": exc.code}
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
        return exc.exit_code
