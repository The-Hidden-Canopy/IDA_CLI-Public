# IDA_CLI-Public engineering boundary

This repository is the public user-facing Ask IDA CLI only.

- Proprietary authority, governance, orchestration, private prompts, private
  corpora, training runtimes, and deployment secrets stay outside this repo.
- The CLI is local-first. Discovery, status, inference, and documentation
  explanation do not contact the network.
- Catalog refresh and model download are explicit operator actions.
- Hub catalog data is metadata and attestation, not a model-byte proxy.
- Downloaded model code is never executed. Remote-code loading is disabled.
- Unsupported experimental models are clearly marked and never represented as
  Hub-approved capability or promotion evidence.
- Do not add workbook/report-generation code or private FamilyEngine imports.

Before release, run the focused tests, the clean-history/provenance scan, and
the public-boundary scan. A passing test suite is not GPU or native-training
proof.

