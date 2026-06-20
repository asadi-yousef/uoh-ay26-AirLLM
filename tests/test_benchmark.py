from pathlib import Path

from airllm_ex05.benchmark import load_results, save_result, save_results_csv
from airllm_ex05.models import BenchmarkResult, MetricSummary


def test_result_serialization_roundtrip(tmp_path: Path) -> None:
    result = BenchmarkResult(
        runner="baseline",
        model_name="model",
        prompt="hello",
        prompt_index=0,
        run_index=0,
        status="success",
        metrics=MetricSummary(output_tokens=2, total_latency_seconds=1.0),
    )

    save_result(result, tmp_path)
    loaded = load_results(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].runner == "baseline"


def test_save_results_csv(tmp_path: Path) -> None:
    result = BenchmarkResult(
        runner="airllm",
        model_name="model",
        prompt="hello",
        prompt_index=0,
        run_index=0,
        status="failed",
        error_message="missing dependency",
    )

    output = save_results_csv([result], tmp_path / "results.csv")

    assert output.exists()
    assert "missing dependency" in output.read_text(encoding="utf-8")
