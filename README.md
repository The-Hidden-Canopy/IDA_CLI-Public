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
```

The network is used only by the explicit `catalog` and `model download`
commands. The public explanation mode searches the bundled public
documentation snapshot and does not call a private service.

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

Release 0.2.1 pins an Ed25519 verification key in `ask_ida_cli/catalog_public_key.pem`.
The Hub deployment must provide the matching private signing key through its
secret configuration; the private key is never stored in this repository. A
missing signing key, unsigned catalog, stale catalog, or unverified artifact
fails closed.
