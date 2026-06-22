from pathlib import Path

from airllm_ex05.cli import main


def test_cli_hardware_smoke(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    assert main(["hardware", "--config", str(config_path)]) == 0
    assert (tmp_path / "results" / "raw" / "hardware.json").exists()


def test_cli_baseline_saves_failed_results(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    raw_dir = tmp_path / "results" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "baseline_p9_r9.json").write_text("stale", encoding="utf-8")
    (raw_dir / "baseline_results.csv").write_text("stale", encoding="utf-8")

    assert main(["baseline", "--config", str(config_path)]) == 0
    assert list(raw_dir.glob("baseline_p*_r*.json"))
    assert not (raw_dir / "baseline_p9_r9.json").exists()
    assert (raw_dir / "baseline_results.csv").exists()
    assert (raw_dir / "baseline_results.csv").read_text(encoding="utf-8") != "stale"


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "experiment.yaml"
    cache_dir = (tmp_path / "cache").as_posix()
    raw_dir = (tmp_path / "results" / "raw").as_posix()
    processed_dir = (tmp_path / "results" / "processed").as_posix()
    figures_dir = (tmp_path / "results" / "figures").as_posix()
    report_path = (tmp_path / "REPORT.md").as_posix()
    airllm_dir = (tmp_path / "airllm").as_posix()
    config_path.write_text(
        f"""
model:
  name: "sshleifer/tiny-gpt2"
  cache_dir: "{cache_dir}"
  trust_remote_code: false
  device: "auto"
benchmark:
  prompts: ["hello"]
  max_new_tokens: 2
  runs: 1
  warmup_runs: 0
  timeout_seconds: 10
outputs:
  raw_dir: "{raw_dir}"
  processed_dir: "{processed_dir}"
  figures_dir: "{figures_dir}"
  report_path: "{report_path}"
baseline:
  enabled: true
  mode: "transformers"
airllm:
  enabled: true
  mode: "airllm"
  layer_shards_saving_path: "{airllm_dir}"
quantization:
  enabled: true
  mode: "bitsandbytes"
  bits: 4
  compute_dtype: "float16"
cost:
  api_input_price_per_1m_tokens_usd: 0.15
  api_output_price_per_1m_tokens_usd: 0.60
  cached_input_discount: 0.5
  hardware_cost_usd: 1200.0
  hardware_lifetime_months: 36
  electricity_usd_per_kwh: 0.20
  average_power_watts: 180.0
  maintenance_monthly_usd: 10.0
  monthly_request_volumes: [100]
""",
        encoding="utf-8",
    )
    return config_path
