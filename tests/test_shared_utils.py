from pathlib import Path

import pytest

from airllm_ex05.shared.paths import ensure_directory, resolve_path
from airllm_ex05.shared.validation import require_file, require_positive_int


def test_ensure_directory_creates_path(tmp_path: Path) -> None:
    target = tmp_path / "nested"

    assert ensure_directory(target) == target
    assert target.is_dir()


def test_resolve_path_keeps_absolute(tmp_path: Path) -> None:
    assert resolve_path(tmp_path) == tmp_path


def test_require_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        require_file(tmp_path / "missing.yaml")


def test_require_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        require_file(tmp_path)


def test_require_positive_int_rejects_zero() -> None:
    with pytest.raises(ValueError):
        require_positive_int(0, "runs")
