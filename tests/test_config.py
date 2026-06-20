from pathlib import Path

from airllm_ex05.config import load_config


def test_load_config_resolves_paths() -> None:
    config = load_config("configs/experiment.yaml")

    assert config.model.name
    assert config.benchmark.max_new_tokens > 0
    assert config.outputs.raw_dir.is_absolute()
    assert config.outputs.raw_dir.name == "raw"


def test_load_config_from_absolute_path() -> None:
    config_path = Path("configs/experiment.yaml").resolve()
    config = load_config(config_path)

    assert config.quantization.bits in {4, 8}
