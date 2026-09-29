# IDA_CLI-Public

The public, local-first Ask IDA command-line interface.

This repository contains the regular user-facing CLI surface. Proprietary Hub
authority and private runtime code remain separate. The CLI can use an
anonymous Hub dependency catalog for reviewed metadata, then resolves approved
model bytes locally from Hugging Face and verifies the resulting artifact.
Hub does not receive model bytes or local paths.

## Public Hub

- [Ask IDA CLI overview](https://thehiddencanopy.com/ask-ida-cli.html)
- [IDA_CLI-Public release notes](https://thehiddencanopy.com/updates.html#release-notes)
- [Support IDA_CLI-Public](https://thehiddencanopy.com/flight-deck.html#support)

## Commands

```text
ida status
ida catalog
ida explain <query>
ida model list
ida model download <catalog-id>
ida model download --experimental <org/model>
ida ask <model-directory-or-id> <prompt>
ida leaderboard list
ida telemetry status
ida telemetry share-kernel <foundry-kernel-telemetry-dir> --url https://example.invalid/telemetry
```

The network is used only by the explicit `catalog` and `model download`
commands, the explicit read-only `leaderboard list` command, and the explicit
`telemetry share-kernel` command. The leaderboard command performs an
anonymous HTTPS GET only; it has no submission, publication, bearer-token, or
background-upload path. The Hub URL can be overridden with `--url`,
`--leaderboard-url`, or `ASK_IDA_PUBLIC_LEADERBOARD_URL`.

Kernel telemetry collection and sharing are disabled by default. Foundry's opt-in
`--kernel-telemetry DIR` output uses the V2-style
`native-kernel-telemetry.v2` receipt with `kernel_ontology.jsonl`,
`training_metrics.jsonl`, and SHA-256 integrity records. The share command
rejects paths, prompts, datasets, checkpoints, model weights, credentials, raw
logs, unknown fields, device-duration claims, and promotion claims. Use
`--dry-run` to inspect the validated receipt without network access. An
optional bearer token is used only when the explicit share command includes
`--send-token` together with an explicit `--url`; it is read from
`ASK_IDA_PUBLIC_TELEMETRY_TOKEN` and is never printed or persisted by the CLI.
No Hub receiver is assumed: provide `--url`, `--telemetry-url`, or
`ASK_IDA_PUBLIC_TELEMETRY_URL` for the explicit POST destination. The CLI does
not upload in the background or retain a copy of the bundle.

Reviewed catalog entries require an immutable revision and content hash.
Public model downloads and inference accept safetensors weights only; legacy
serialized weight files are rejected.
Experimental model IDs are allowed only from the configured public host
allowlist, initially Hugging Face. They are not supported capability claims;
support is provided only when the maintainers can reproduce the request.

Install the base CLI with the normal Python package tool. Local inference and
downloads use the optional runtime extra:

```text
python -m pip install -e ".[runtime]"
ida status
```

`ida ask` accepts either a complete local model directory or the name of a
complete directory beneath the configured public model root. It never resolves
an ask target over the network.

Release 0.2.1 pins an Ed25519 verification key in `ask_ida_cli/catalog_public_key.pem`.
The Hub deployment must provide the matching private signing key through its
secret configuration; the private key is never stored in this repository. A
missing signing key, unsigned catalog, stale catalog, or unverified artifact
fails closed.
