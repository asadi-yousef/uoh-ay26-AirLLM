"""Quantized Hugging Face runner."""

import importlib
import time
from typing import Any

from airllm_ex05.config import ExperimentConfig
from airllm_ex05.constants import RUNNER_QUANTIZED
from airllm_ex05.models import BenchmarkResult
from airllm_ex05.runners.baseline_runner import _generate
from airllm_ex05.runners.common import failed_result, iter_prompts, run_prompt


def run_quantized(config: ExperimentConfig) -> list[BenchmarkResult]:
    """Run transformers quantized inference or return structured failures."""
    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        model, tokenizer, load_time, method = _load_quantized(config, transformers, torch)
    except Exception as exc:
        return [
            failed_result(config, RUNNER_QUANTIZED, prompt, pi, ri, exc, {"stage": "load"})
            for prompt, pi, ri in iter_prompts(config)
        ]

    results = []
    for prompt, prompt_index, run_index in iter_prompts(config):
        try:
            results.append(
                run_prompt(
                    config,
                    RUNNER_QUANTIZED,
                    prompt,
                    prompt_index,
                    run_index,
                    lambda prompt=prompt: _generate(
                        model, tokenizer, prompt, config.benchmark.max_new_tokens
                    ),
                    load_time,
                    {"quantization_method": method, "bits": config.quantization.bits},
                )
            )
        except Exception as exc:
            results.append(
                failed_result(config, RUNNER_QUANTIZED, prompt, prompt_index, run_index, exc)
            )
    return results


def _load_quantized(
    config: ExperimentConfig, transformers: Any, torch: Any
) -> tuple[Any, Any, float, str]:
    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.name,
        cache_dir=str(config.model.cache_dir),
        trust_remote_code=config.model.trust_remote_code,
    )
    kwargs = _quantization_kwargs(config, transformers, torch)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=str(config.model.cache_dir),
        trust_remote_code=config.model.trust_remote_code,
        device_map=config.model.device,
        **kwargs,
    )
    model.eval()
    return model, tokenizer, time.perf_counter() - started, str(kwargs)


def _quantization_kwargs(config: ExperimentConfig, transformers: Any, torch: Any) -> dict[str, Any]:
    if not hasattr(transformers, "BitsAndBytesConfig"):
        return {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
        }
    if config.quantization.bits == 4:
        return {"quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True)}
    return {"quantization_config": transformers.BitsAndBytesConfig(load_in_8bit=True)}
