from airllm_ex05.config import load_config
from airllm_ex05.runners.airllm_runner import run_airllm
from airllm_ex05.runners.baseline_runner import run_baseline
from airllm_ex05.runners.quantized_runner import run_quantized
from fake_model_deps import fake_import, missing_transformers_import


def test_baseline_missing_transformers_returns_failures(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", missing_transformers_import)
    config = load_config("configs/experiment.yaml")
    results = run_baseline(config)

    assert results
    assert all(result.status == "failed" for result in results)
    assert all(result.metadata["stage"] == "load" for result in results)


def test_baseline_success_with_fakes(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", fake_import)
    config = load_config("configs/experiment.yaml")

    results = run_baseline(config)

    assert all(result.status == "success" for result in results)
    assert results[0].generated_text == "generated text"


def test_airllm_missing_dependency_returns_failures() -> None:
    config = load_config("configs/experiment.yaml")
    results = run_airllm(config)

    assert results
    assert all(result.runner == "airllm" for result in results)
    assert all(result.status == "failed" for result in results)


def test_airllm_success_with_fakes(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", fake_import)
    config = load_config("configs/experiment.yaml")

    results = run_airllm(config)

    assert all(result.status == "success" for result in results)
    assert "layer_shards" in results[0].metadata


def test_quantized_missing_transformers_returns_failures(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", missing_transformers_import)
    config = load_config("configs/experiment.yaml")
    results = run_quantized(config)

    assert results
    assert all(result.runner == "quantized" for result in results)
    assert all(result.status == "failed" for result in results)


def test_quantized_success_with_fakes(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", fake_import)
    config = load_config("configs/experiment.yaml")

    results = run_quantized(config)

    assert all(result.status == "success" for result in results)
    assert results[0].metadata["bits"] == 8
    assert "quantization_config" in results[0].metadata["quantization_method"]
