# Public boundary

`IDA_CLI-Public` is the regular user-facing surface. It is intentionally not
the Hub authority plane and not the IDA training runtime.

The Hub publishes a signed dependency catalog containing public metadata and
attestations. The CLI resolves bytes locally and records a local redacted
receipt. Model bytes, local paths, worker tokens, organization scope, private
prompts, and private runtime references do not cross the boundary.

The public explanation command searches the bundled public documentation
snapshot. Workbook generation and private explanatory systems are not part of
this repository.

The CLI has three explicit public network paths: catalog refresh and model
download, anonymous read-only leaderboard queries, and an operator-invoked
kernel-telemetry share. None runs in the background. Local status, model
listing, inference, and documentation explanation remain offline.

## Public leaderboard read path

`ida leaderboard list` is an explicit, anonymous HTTPS GET to the Hub's
confirmation-gated Neural Forge leaderboard. The CLI validates the
`neural-forge-leaderboard.v1` envelope, rejects private or malformed fields,
and returns only the public model/revision, sanitized GPU, pseudonymous
operator, approved result, and publication timestamp. It has no run-submission,
publication-confirmation, evaluator, worker-token, or background-upload path.
