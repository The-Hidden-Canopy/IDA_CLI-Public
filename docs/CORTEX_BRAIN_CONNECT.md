# Cortex local brain connector candidate

This public-safe transport contains no numerical runtime, private IDA imports,
checkpoint reader, model weights, downloaded Python or training implementation.
It does not require a private source checkout. It connects to an independently
installed, owner-controlled local IDA runtime that has already validated and
loaded the brain. Installing this connector alone does not make arbitrary
safetensors or a custom architecture runnable.

The release candidate is not a signed or published installer. Independent
review, clean-machine real-runtime acceptance and public distribution approval
remain open. Do not claim that this source closes Cortex release gate G07.

Build a wheel using `python -m pip wheel --no-deps . --wheel-dir DIST`, then
install that exact reviewed wheel in a fresh virtual environment. Record its
SHA-256. Do not install the public and private packages in one environment:
they share the `ask_ida_cli` module name and the `ida` executable name.

```text
ida brain connect --runtime-url http://127.0.0.1:8780 --username YOUR_USERNAME --title "My local brain" --approve --justification "Share my local brain through Cortex"
```

The Cortex password is prompted, never accepted as an argument, printed or
persisted. Only the explicit connect command uses the network for this flow.
HTTPS goes only to cortexcommons.com; redirects are refused. The local runtime
must use numeric loopback with an explicit port. The Cortex session is never
sent to the local runtime. Ctrl+C disconnects and revokes only this CLI session.
Killing the process relies on server heartbeat/session expiry instead.

The installed runtime must expose this bounded local bridge:

- `GET /cortex/metadata`: exactly artifact_sha256, artifact_format,
  storage_precision and master_format, after identity validation/model loading.
- `POST /cortex/complete`: exactly prompt and max_new_bytes. Start fresh state
  per request. Return the documented eight-field IDA completion receipt.
- Supported server formats are the explicit CPU byte package and regional MXFP8
  adapter. Regional stored precision is MXFP8; current compute is decoded FP32.
  Hardware/output receipts remain runner-reported, not remotely attested.

No Cortex response can select an executable, command, file, source or runtime
module. Weights stay local. Prompts and bounded results pass through Cortex for
sharing. Keep this process and its runtime alive to remain online.

Windows: create a venv with `py -3 -m venv .venv` and invoke
`.venv\Scripts\python -m pip install PATH_TO_REVIEWED_WHEEL`. Linux/macOS: use
`python3 -m venv .venv` and `.venv/bin/python -m pip install PATH_TO_REVIEWED_WHEEL`.
To uninstall, run that venv's `python -m pip uninstall ida-cli-public`. The
connector creates no credential file. Offline runtime, unsupported metadata,
expired Cortex session and redirects fail with redacted stable errors; configure
the owner runtime separately rather than adding private code to this package.
