"""Quantized Hugging Face runner."""

import importlib
import re
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
    if config.quantization.mode == "dynamic_int8":
        _raise_if_dynamic_int8_too_large(config)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name,
            cache_dir=str(config.model.cache_dir),
            trust_remote_code=config.model.trust_remote_code,
            torch_dtype=_torch_dtype(config, torch),
            low_cpu_mem_usage=True,
        )
        model.eval()
        quantize_dynamic = torch.ao.quantization.quantize_dynamic
        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        return model, tokenizer, time.perf_counter() - started, "torch.dynamic_int8"

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
            "torch_dtype": _torch_dtype(config, torch),
            "low_cpu_mem_usage": True,
        }
    if config.quantization.bits == 4:
        return {"quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True)}
    return {
        "quantization_config": transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    }


def _torch_dtype(config: ExperimentConfig, torch: Any) -> Any:
    dtype_name = config.quantization.compute_dtype
    return getattr(torch, dtype_name, torch.float16)


def _raise_if_dynamic_int8_too_large(config: ExperimentConfig) -> None:
    checkpoint_bytes = _cached_checkpoint_size(config)
    if checkpoint_bytes == 0:
        return
    try:
        psutil = importlib.import_module("psutil")
    except ImportError:
        return
    total_ram_bytes = psutil.virtual_memory().total
    required_bytes = checkpoint_bytes * 2.25
    if total_ram_bytes >= required_bytes:
        return
    checkpoint_gb = checkpoint_bytes / 1024**3
    total_ram_gb = total_ram_bytes / 1024**3
    required_gb = required_bytes / 1024**3
    msg = (
        "dynamic_int8 quantization would likely exceed local RAM before results can be "
        f"serialized: cached checkpoint is {checkpoint_gb:.1f} GiB, physical RAM is "
        f"{total_ram_gb:.1f} GiB, estimated requirement is {required_gb:.1f} GiB. "
        "Use a smaller model, a native low-bit backend such as bitsandbytes on a supported "
        "CUDA/Linux environment, or preserve this structured failure as the 7B quantized result."
    )
    raise MemoryError(msg)


def _cached_checkpoint_size(config: ExperimentConfig) -> int:
    cache_name = "models--" + re.sub(r"[\\/]", "--", config.model.name)
    model_cache = config.model.cache_dir / cache_name
    if not model_cache.exists():
        return 0
    return sum(path.stat().st_size for path in model_cache.rglob("*.safetensors"))
