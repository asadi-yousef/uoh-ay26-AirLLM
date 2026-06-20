"""Shared runner helpers."""

import time
from collections.abc import Callable

from airllm_ex05.config import ExperimentConfig
from airllm_ex05.metrics import (
    MemorySampler,
    count_simple_tokens,
    cuda_peak_memory_mb,
    time_per_token,
    tokens_per_second,
)
from airllm_ex05.models import BenchmarkResult, MetricSummary


def failed_result(
    config: ExperimentConfig,
    runner: str,
    prompt: str,
    prompt_index: int,
    run_index: int,
    error: BaseException,
    metadata: dict[str, object] | None = None,
) -> BenchmarkResult:
    """Create a structured failed result."""
    return BenchmarkResult(
        runner=runner,
        model_name=config.model.name,
        prompt=prompt,
        prompt_index=prompt_index,
        run_index=run_index,
        status="failed",
        error_type=type(error).__name__,
        error_message=str(error),
        metadata=metadata or {},
    )


def run_prompt(
    config: ExperimentConfig,
    runner: str,
    prompt: str,
    prompt_index: int,
    run_index: int,
    generation: Callable[[], str],
    load_time_seconds: float | None,
    metadata: dict[str, object] | None = None,
) -> BenchmarkResult:
    """Measure one prompt generation call."""
    sampler = MemorySampler()
    started = time.perf_counter()
    generated = sampler.run(generation)
    latency = time.perf_counter() - started
    output_tokens = count_simple_tokens(generated)
    return BenchmarkResult(
        runner=runner,
        model_name=config.model.name,
        prompt=prompt,
        prompt_index=prompt_index,
        run_index=run_index,
        status="success",
        generated_text=generated[:1000],
        metadata=metadata or {},
        metrics=MetricSummary(
            load_time_seconds=load_time_seconds,
            total_latency_seconds=latency,
            time_per_output_token_seconds=time_per_token(output_tokens, latency),
            tokens_per_second=tokens_per_second(output_tokens, latency),
            input_tokens=count_simple_tokens(prompt),
            output_tokens=output_tokens,
            peak_ram_mb=sampler.peak_ram_mb,
            peak_vram_mb=cuda_peak_memory_mb(),
        ),
    )


def iter_prompts(config: ExperimentConfig) -> list[tuple[str, int, int]]:
    """Return prompt/run tuples for configured benchmark runs."""
    return [
        (prompt, prompt_index, run_index)
        for prompt_index, prompt in enumerate(config.benchmark.prompts)
        for run_index in range(config.benchmark.runs)
    ]
