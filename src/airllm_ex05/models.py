"""Data models for benchmark results."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["success", "failed"]


class MetricSummary(BaseModel):
    """Measured performance metrics for one inference attempt."""

    load_time_seconds: float | None = None
    total_latency_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    time_per_output_token_seconds: float | None = None
    tokens_per_second: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None


class BenchmarkResult(BaseModel):
    """Serializable result from a single runner attempt."""

    runner: str
    model_name: str
    prompt: str
    prompt_index: int
    run_index: int
    status: Status
    metrics: MetricSummary = Field(default_factory=MetricSummary)
    generated_text: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def filename(self) -> str:
        """Return a stable filename for raw result storage."""
        return f"{self.runner}_p{self.prompt_index}_r{self.run_index}.json"


class HardwareInfo(BaseModel):
    """Hardware and runtime environment information."""

    cpu_model: str
    physical_cores: int | None
    logical_cores: int
    ram_total_gb: float
    gpu_model: str | None
    vram_total_gb: float | None
    cuda_available: bool
    os: str
    python_version: str
    storage: list[dict[str, Any]] = Field(default_factory=list)


def write_json_model(model: BaseModel, path: Path) -> Path:
    """Write a Pydantic model as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return path
