"""Path helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root from the installed source layout."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a path relative to `base_dir` or the project root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or project_root()).joinpath(path).resolve()


def ensure_directory(path: Path) -> Path:
    """Create a directory when needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
