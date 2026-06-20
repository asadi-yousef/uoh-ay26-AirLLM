"""Experiment configuration loading."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from airllm_ex05.shared.paths import resolve_path
from airllm_ex05.shared.validation import require_file, require_positive_int


class OutputConfig(BaseModel):
    """Output directory configuration."""

    raw_dir: Path
    processed_dir: Path
    figures_dir: Path
    report_path: Path


class ModelConfig(BaseModel):
    """Model selection configuration."""

    name: str
    cache_dir: Path
    trust_remote_code: bool = False
    device: str = "auto"


class BenchmarkConfig(BaseModel):
    """Benchmark execution settings."""

    prompts: list[str] = Field(min_length=1)
    max_new_tokens: int = 32
    runs: int = 1
    warmup_runs: int = 0
    timeout_seconds: int = 900


class RunnerConfig(BaseModel):
    """Generic runner settings."""

    enabled: bool = True
    mode: str = "default"


class AirLLMConfig(RunnerConfig):
    """AirLLM-specific settings."""

    layer_shards_saving_path: Path


class QuantizationConfig(RunnerConfig):
    """Quantization-specific settings."""

    bits: Literal[4, 8] = 4
    compute_dtype: str = "float16"


class CostConfig(BaseModel):
    """Economic-analysis assumptions."""

    api_input_price_per_1m_tokens_usd: float = 0.15
    api_output_price_per_1m_tokens_usd: float = 0.60
    cached_input_discount: float = 0.5
    hardware_cost_usd: float = 1200.0
    hardware_lifetime_months: int = 36
    electricity_usd_per_kwh: float = 0.20
    average_power_watts: float = 180.0
    maintenance_monthly_usd: float = 10.0
    monthly_request_volumes: list[int] = Field(default_factory=lambda: [100, 1000, 10000])


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    model: ModelConfig
    benchmark: BenchmarkConfig
    outputs: OutputConfig
    baseline: RunnerConfig
    airllm: AirLLMConfig
    quantization: QuantizationConfig
    cost: CostConfig


def _resolve_paths(data: dict, config_path: Path) -> dict:
    base_dir = config_path.parent.parent
    for key in ("raw_dir", "processed_dir", "figures_dir", "report_path"):
        data["outputs"][key] = resolve_path(data["outputs"][key], base_dir)
    data["model"]["cache_dir"] = resolve_path(data["model"]["cache_dir"], base_dir)
    data["airllm"]["layer_shards_saving_path"] = resolve_path(
        data["airllm"]["layer_shards_saving_path"], base_dir
    )
    return data


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML."""
    path = require_file(resolve_path(config_path))
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    config = ExperimentConfig.model_validate(_resolve_paths(data, path))
    require_positive_int(config.benchmark.max_new_tokens, "max_new_tokens")
    require_positive_int(config.benchmark.runs, "runs")
    return config
