import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from airllm_ex05.config import load_config
from airllm_ex05.runners.quantized_runner import (
    _cached_checkpoint_size,
    _raise_if_dynamic_int8_too_large,
)


def test_cached_checkpoint_size_uses_huggingface_cache_layout(tmp_path: Path) -> None:
    config = load_config("configs/experiment.yaml")
    config.model.name = "Org/Test-7B"
    config.model.cache_dir = tmp_path
    model_cache = tmp_path / "models--Org--Test-7B" / "snapshots" / "abc"
    model_cache.mkdir(parents=True)
    (model_cache / "model.safetensors").write_bytes(b"1234")
    (model_cache / "config.json").write_text("{}", encoding="utf-8")

    assert _cached_checkpoint_size(config) == 4


def test_dynamic_int8_memory_guard_allows_small_cache(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_config("configs/experiment.yaml")
    config.model.name = "Org/Test-7B"
    config.model.cache_dir = tmp_path
    model_cache = tmp_path / "models--Org--Test-7B"
    model_cache.mkdir()
    (model_cache / "model.safetensors").write_bytes(b"1234")
    _fake_psutil(monkeypatch, total=1024)

    _raise_if_dynamic_int8_too_large(config)


def test_dynamic_int8_memory_guard_raises_for_oversized_cache(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_config("configs/experiment.yaml")
    config.model.name = "Org/Test-7B"
    config.model.cache_dir = tmp_path
    model_cache = tmp_path / "models--Org--Test-7B"
    model_cache.mkdir()
    (model_cache / "model.safetensors").write_bytes(b"1234")
    _fake_psutil(monkeypatch, total=8)

    with pytest.raises(MemoryError, match="dynamic_int8 quantization"):
        _raise_if_dynamic_int8_too_large(config)


def _fake_psutil(monkeypatch, total: int) -> None:
    original_import_module = importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "psutil":
            return SimpleNamespace(
                virtual_memory=lambda: SimpleNamespace(total=total)
            )
        return original_import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)
