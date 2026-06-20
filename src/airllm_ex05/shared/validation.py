"""Validation helpers."""

from pathlib import Path


def require_file(path: Path) -> Path:
    """Return `path` if it exists, otherwise raise a clear error."""
    if not path.exists():
        msg = f"Required file does not exist: {path}"
        raise FileNotFoundError(msg)
    if not path.is_file():
        msg = f"Expected a file, got: {path}"
        raise ValueError(msg)
    return path


def require_positive_int(value: int, name: str) -> int:
    """Validate a positive integer setting."""
    if value < 1:
        msg = f"{name} must be >= 1, got {value}"
        raise ValueError(msg)
    return value
