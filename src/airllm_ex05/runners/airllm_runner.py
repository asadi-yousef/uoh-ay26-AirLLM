"""AirLLM runner."""

import importlib
import time
from typing import Any

from airllm_ex05.config import ExperimentConfig
from airllm_ex05.constants import RUNNER_AIRLLM
from airllm_ex05.models import BenchmarkResult
from airllm_ex05.runners.common import failed_result, iter_prompts, run_prompt


def run_airllm(config: ExperimentConfig) -> list[BenchmarkResult]:
    """Run AirLLM inference or return structured failures."""
    try:
        airllm = importlib.import_module("airllm")
        model, load_time = _load_airllm_model(config, airllm)
    except Exception as exc:
        return [
            failed_result(config, RUNNER_AIRLLM, prompt, pi, ri, exc, {"stage": "load"})
            for prompt, pi, ri in iter_prompts(config)
        ]

    results = []
    for prompt, prompt_index, run_index in iter_prompts(config):
        try:
            results.append(
                run_prompt(
                    config,
                    RUNNER_AIRLLM,
                    prompt,
                    prompt_index,
                    run_index,
                    lambda prompt=prompt: _generate(model, prompt, config.benchmark.max_new_tokens),
                    load_time,
                    {"layer_shards": str(config.airllm.layer_shards_saving_path)},
                )
            )
        except Exception as exc:
            results.append(failed_result(config, RUNNER_AIRLLM, prompt, prompt_index, run_index, exc))
    return results


def _load_airllm_model(config: ExperimentConfig, airllm: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    model_class = getattr(airllm, "AutoModel", None) or getattr(airllm, "AirLLMLlama2", None)
    if model_class is None:
        msg = "Installed airllm package does not expose AutoModel or AirLLMLlama2."
        raise RuntimeError(msg)
    model = model_class.from_pretrained(
        config.model.name,
        layer_shards_saving_path=str(config.airllm.layer_shards_saving_path),
    )
    return model, time.perf_counter() - started


def _generate(model: Any, prompt: str, max_new_tokens: int) -> str:
    output = model.generate(prompt, max_new_tokens=max_new_tokens)
    if isinstance(output, list):
        return str(output[0])
    return str(output)
