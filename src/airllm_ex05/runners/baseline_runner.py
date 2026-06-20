"""Direct Hugging Face baseline runner."""

import importlib
import threading
import time
from typing import Any

from airllm_ex05.config import ExperimentConfig
from airllm_ex05.constants import RUNNER_BASELINE
from airllm_ex05.models import BenchmarkResult
from airllm_ex05.runners.common import GeneratedOutput, failed_result, iter_prompts, run_prompt


def run_baseline(config: ExperimentConfig) -> list[BenchmarkResult]:
    """Run direct transformers inference or return structured failures."""
    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        model, tokenizer, load_time = _load_model(config, transformers, torch)
    except Exception as exc:
        return [
            failed_result(config, RUNNER_BASELINE, prompt, pi, ri, exc, {"stage": "load"})
            for prompt, pi, ri in iter_prompts(config)
        ]

    results = []
    for prompt, prompt_index, run_index in iter_prompts(config):
        try:
            results.append(
                run_prompt(
                    config,
                    RUNNER_BASELINE,
                    prompt,
                    prompt_index,
                    run_index,
                    lambda prompt=prompt: _generate(
                        model, tokenizer, prompt, config.benchmark.max_new_tokens
                    ),
                    load_time,
                    {"mode": config.baseline.mode},
                )
            )
        except Exception as exc:
            results.append(
                failed_result(config, RUNNER_BASELINE, prompt, prompt_index, run_index, exc)
            )
    return results


def _load_model(config: ExperimentConfig, transformers: Any, torch: Any) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.name,
        cache_dir=str(config.model.cache_dir),
        trust_remote_code=config.model.trust_remote_code,
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=str(config.model.cache_dir),
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=dtype,
        device_map=config.model.device,
    )
    model.eval()
    return model, tokenizer, time.perf_counter() - started


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> GeneratedOutput:
    streamer_class = getattr(importlib.import_module("transformers"), "TextIteratorStreamer", None)
    if streamer_class is not None:
        streamed = _generate_streamed(model, tokenizer, streamer_class, prompt, max_new_tokens)
        if streamed is not None:
            return streamed

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _generate_streamed(
    model: Any,
    tokenizer: Any,
    streamer_class: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float | None] | None:
    try:
        streamer = streamer_class(tokenizer, skip_prompt=True, skip_special_tokens=True)
    except TypeError:
        return None
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    generated_parts = []
    error: list[BaseException] = []
    started = time.perf_counter()
    thread = threading.Thread(
        target=_generate_in_thread,
        args=(model, inputs, streamer, max_new_tokens, error),
        daemon=True,
    )
    thread.start()
    first_token_time = None
    for chunk in streamer:
        if chunk and first_token_time is None:
            first_token_time = time.perf_counter() - started
        generated_parts.append(chunk)
    thread.join()
    if error:
        raise error[0]
    return prompt + "".join(generated_parts), first_token_time


def _generate_in_thread(
    model: Any,
    inputs: dict[str, Any],
    streamer: Any,
    max_new_tokens: int,
    error: list[BaseException],
) -> None:
    try:
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )
    except BaseException as exc:
        error.append(exc)
