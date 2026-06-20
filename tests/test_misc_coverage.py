from pathlib import Path

from airllm_ex05 import constants
from airllm_ex05.benchmark import load_results
from airllm_ex05.metrics import MemorySampler, current_process_ram_mb, cuda_peak_memory_mb
from airllm_ex05.shared.logging_utils import configure_logging


def test_constants_are_defined() -> None:
    assert constants.DEFAULT_CONFIG_PATH.endswith(".yaml")
    assert constants.RUNNER_BASELINE == "baseline"


def test_configure_logging() -> None:
    configure_logging(verbose=True)


def test_memory_helpers_do_not_fail() -> None:
    assert current_process_ram_mb() > 0
    assert cuda_peak_memory_mb() is None or cuda_peak_memory_mb() >= 0


def test_memory_sampler_runs_function() -> None:
    sampler = MemorySampler(interval_seconds=0.001)

    assert sampler.run(lambda: "ok") == "ok"
    assert sampler.peak_ram_mb > 0


def test_load_results_empty_directory(tmp_path: Path) -> None:
    assert load_results(tmp_path) == []
