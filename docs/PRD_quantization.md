# Quantization PRD

## Purpose

Attempt lower-precision inference so the project can compare memory, latency, throughput, and
output quality against direct baseline and AirLLM execution when generation succeeds, while still
preserving structured failures for backend or memory-limited configurations.

## Context

Quantization reduces the precision of model weights or operations. In local inference this can
reduce memory footprint and memory bandwidth pressure, especially during decode. The tradeoff
is backend compatibility and possible output-quality degradation.

## Functional Requirements

- Read quantization mode from config.
- Read bit width from config.
- Read compute dtype from config.
- Support `bitsandbytes` 4-bit configuration.
- Support `bitsandbytes` 8-bit configuration.
- Support CPU `dynamic_int8`.
- Load tokenizer and model through Transformers.
- Apply dynamic quantization with `torch.ao.quantization.quantize_dynamic`.
- Reuse baseline generation where possible.
- Measure load time, latency, TTFT, TPOT, throughput, RAM, and VRAM where available.
- Save generated output.
- Save structured failures when a backend is unavailable or likely to exceed local RAM.
- Include quantization metadata in successful results.
- Include load-stage metadata in failure results.
- Detect cached checkpoint footprint for CPU dynamic-int8.
- Avoid native memory crashes when a 7B dynamic-int8 conversion is likely to exceed RAM.

## Non-Functional Requirements

- Tests must not require `bitsandbytes`.
- Tests must not download models.
- Windows and CPU quantization limitations must be documented.
- The report must not claim universal memory reduction unless evidence shows it.
- The runner must be able to fail without stopping analysis.
- Stale quantized JSON/CSV outputs must not survive a fresh run.

## Acceptance Criteria

- `uv run airllm-ex05 quantized --config configs/experiment.yaml` writes one result per
  prompt/run.
- `bitsandbytes` failures are clear when that mode is selected.
- CPU dynamic-int8 is implemented and tested for local validation.
- Successful rows include quantization metadata when generation succeeds.
- Failure rows include `stage=load` when loading or conversion is stopped before generation.
- Final report compares quantized results against baseline only for metrics that exist.
- Final report includes quantized latency, throughput, memory, and quality notes when generation
  succeeds.

## Final Observed Evidence

- Final quantization mode: `bitsandbytes`.
- Final quantization bits: 8.
- CPU offload is enabled through `llm_int8_enable_fp32_cpu_offload=True`.
- Both final quantized prompt rows succeeded.
- Quantized latency, TTFT, TPOT, throughput, RAM, VRAM, and output quality smoke-check evidence
  are available for the 7B run.
- CPU `dynamic_int8` remains implemented and tested for smaller CPU validation models.
- The previous 7B dynamic-int8 attempt was rejected because conversion was likely to exceed the
  observed 15.7 GiB RAM budget.

## Risks

- `bitsandbytes` may not support the current OS, Python version, or hardware.
- CUDA-oriented quantization paths may fail on CPU-only machines.
- Some models may not support a chosen quantization configuration.
- Dynamic-int8 can increase load time.
- Dynamic-int8 can temporarily need more RAM than the final compressed model size.
- Aggressive quantization can harm output quality.
- Small prompt samples cannot prove broad quality behavior.

## Implementation Notes

- Main file: `src/airllm_ex05/runners/quantized_runner.py`
- Shared generation helper: `src/airllm_ex05/runners/baseline_runner.py`
- Metrics: `src/airllm_ex05/metrics.py`
- Runner tests: `tests/test_runners.py`
- Memory guard tests: `tests/test_quantized_guard.py`

## Report Requirements

The report must separate these claims:

- Quantized runner implementation exists and is tested.
- bitsandbytes 8-bit with CPU offload is the configured final 7B backend.
- Dynamic-int8 remains available for smaller CPU validation models.
- Final 7B bitsandbytes inference succeeded on the observed laptop.
- The final evidence can compare quantized speed, memory, and output quality smoke checks against
  baseline and AirLLM.
