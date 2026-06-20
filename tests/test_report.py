from pathlib import Path

from airllm_ex05.config import load_config
from airllm_ex05.models import BenchmarkResult, MetricSummary, write_json_model
from airllm_ex05.report import analyze_results, generate_report


def test_analyze_and_generate_report(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    result = BenchmarkResult(
        runner="baseline",
        model_name="model",
        prompt="hello",
        prompt_index=0,
        run_index=0,
        status="success",
        metrics=MetricSummary(input_tokens=1, output_tokens=2, total_latency_seconds=1.0),
    )
    write_json_model(result, config.outputs.raw_dir / result.filename())

    analysis_path = analyze_results(config)
    report_path = generate_report(config)

    assert analysis_path.exists()
    assert report_path.exists()
    assert "Prefill" in report_path.read_text(encoding="utf-8")


def _write_config(tmp_path: Path) -> Path:
    source = Path("configs/experiment.yaml").read_text(encoding="utf-8")
    source = source.replace("results/raw", (tmp_path / "raw").as_posix())
    source = source.replace("results/processed", (tmp_path / "processed").as_posix())
    source = source.replace("results/figures", (tmp_path / "figures").as_posix())
    source = source.replace("docs/REPORT.md", (tmp_path / "REPORT.md").as_posix())
    source = source.replace("model_cache/huggingface", (tmp_path / "cache").as_posix())
    source = source.replace("airllm_cache/layer_shards", (tmp_path / "airllm").as_posix())
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(source, encoding="utf-8")
    return config_path
