"""Optional local-only Transformers inference."""

from __future__ import annotations

from pathlib import Path

from .errors import CLIError


def ask_local(model_path: Path, prompt: str, *, max_new_tokens: int = 256) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise CLIError(
            "local inference requires the runtime extra: pip install 'ida-cli-public[runtime]'",
            code="inference_dependency",
            exit_code=3,
        ) from None
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise CLIError("model path is not a complete local model directory", code="model_unavailable", exit_code=3)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype="auto",
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max(1, min(max_new_tokens, 2048)))
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as exc:
        raise CLIError(f"local inference failed: {exc}", code="inference_failed", exit_code=3) from None

