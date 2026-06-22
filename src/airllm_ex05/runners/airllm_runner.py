"""AirLLM runner."""

import importlib
import os
import time
from typing import Any

from airllm_ex05.config import ExperimentConfig
from airllm_ex05.constants import RUNNER_AIRLLM
from airllm_ex05.models import BenchmarkResult
from airllm_ex05.runners.common import failed_result, iter_prompts, run_prompt
from airllm_ex05.shared.paths import ensure_directory


def run_airllm(config: ExperimentConfig) -> list[BenchmarkResult]:
    """Run AirLLM inference or return structured failures."""
    try:
        airllm = importlib.import_module("airllm")
        transformers = importlib.import_module("transformers")
        model, tokenizer, load_time = _load_airllm_model(config, airllm, transformers)
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
                    lambda prompt=prompt: _generate(
                        model, tokenizer, prompt, config.benchmark.max_new_tokens
                    ),
                    load_time,
                    {"layer_shards": str(config.airllm.layer_shards_saving_path)},
                )
            )
        except Exception as exc:
            results.append(
                failed_result(config, RUNNER_AIRLLM, prompt, prompt_index, run_index, exc)
            )
    return results


def _load_airllm_model(
    config: ExperimentConfig, airllm: Any, transformers: Any
) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    cache_dir = str(config.model.cache_dir)
    layer_shards_path = ensure_directory(config.airllm.layer_shards_saving_path)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    model_class = getattr(airllm, "AutoModel", None) or getattr(
        airllm, "AirLLMLlama2", None
    )
    if model_class is None:
        msg = "Installed airllm package does not expose AutoModel or AirLLMLlama2."
        raise RuntimeError(msg)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.name,
        cache_dir=cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = model_class.from_pretrained(
        config.model.name,
        layer_shards_saving_path=str(layer_shards_path),
    )
    return model, tokenizer, time.perf_counter() - started


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)
