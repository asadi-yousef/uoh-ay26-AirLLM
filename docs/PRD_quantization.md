# Quantization PRD

## Purpose

Attempt lower-precision inference so the project can compare memory, latency, throughput, and
output quality against direct baseline and AirLLM execution when generation succeeds, and so it
can preserve a structured failure when the selected model/backend/hardware combination is too
large.

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
- Final report explicitly says when quantized latency, throughput, and quality are unavailable.

## Final Observed Evidence

- Final quantization mode: `dynamic_int8`.
- Both final quantized prompt rows failed before generation.
- Error type: `MemoryError`.
- Error message records that dynamic-int8 quantization would likely exceed local RAM before
  results can be serialized.
- Cached checkpoint estimate in the failure message: about 14.2 GiB.
- Physical RAM estimate in the failure message: about 15.7 GiB.
- Estimated dynamic-int8 conversion requirement: about 31.9 GiB.
- Current processed table has no quantized latency, TTFT, TPOT, throughput, RAM, VRAM, or output
  quality metrics for the 7B run.
- Current processed table supports the claim that the quantized runner is robust and auditable,
  not that final 7B quantized inference succeeded on this laptop.
- Earlier 4-bit `bitsandbytes` path was not suitable for the observed Windows setup.

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
- Dynamic-int8 is the configured Windows CPU backend.
- Final 7B dynamic-int8 inference did not succeed on the observed laptop.
- The failure is structured evidence caused by estimated memory pressure, not a missing run.
- The experiment cannot claim quantized 7B speed, memory, or quality improvements for the final
  evidence.
