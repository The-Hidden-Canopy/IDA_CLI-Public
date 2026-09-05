"""Public CLI errors with stable machine-readable codes."""

from __future__ import annotations


class CLIError(Exception):
    """An expected, user-facing failure."""

    def __init__(self, message: str, *, code: str = "cli_error", exit_code: int = 2) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.exit_code = exit_code

