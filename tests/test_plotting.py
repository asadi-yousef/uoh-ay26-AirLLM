from pathlib import Path

from airllm_ex05.models import BenchmarkResult, MetricSummary
from airllm_ex05.plotting import plot_metric


def test_plot_metric_keeps_failed_runner_without_metric(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            runner="baseline",
            model_name="model",
            prompt="prompt",
            prompt_index=0,
            run_index=0,
            status="success",
            metrics=MetricSummary(total_latency_seconds=1.0),
        ),
        BenchmarkResult(
            runner="quantized",
            model_name="model",
            prompt="prompt",
            prompt_index=0,
            run_index=0,
            status="failed",
            metrics=MetricSummary(),
            error_type="MemoryError",
            error_message="too large",
        ),
    ]

    output = plot_metric(results, "total_latency_seconds", tmp_path / "latency.png")

    assert output is not None
    assert output.exists()
