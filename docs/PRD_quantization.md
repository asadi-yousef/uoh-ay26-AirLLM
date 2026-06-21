# Quantization PRD

## Purpose

Attempt lower-precision inference so the project can compare memory, latency, throughput, and
output quality against direct baseline and AirLLM execution.

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
- Save structured failures when a backend is unavailable.
- Include quantization metadata in successful results.

## Non-Functional Requirements

- Tests must not require `bitsandbytes`.
- Tests must not download models.
- Windows and CPU quantization limitations must be documented.
- The report must not claim universal memory reduction unless evidence shows it.
- The runner must be able to fail without stopping analysis.

## Acceptance Criteria

- `uv run airllm-ex05 quantized --config configs/experiment.yaml` writes one result per
  prompt/run.
- `bitsandbytes` failures are clear when that mode is selected.
- CPU dynamic-int8 works for the final local validation.
- Successful rows include quantization metadata.
- Final report compares quantized results against baseline.

## Final Observed Evidence

- Final quantization mode: `dynamic_int8`.
- Both final prompts succeeded.
- Current processed table shows quantized latency below baseline latency for both prompts.
- Current processed table shows quantized throughput above baseline throughput for both prompts.
- Current processed table does not justify a universal memory-reduction claim.
- Earlier 4-bit `bitsandbytes` path was not suitable for the observed Windows setup.

## Risks

- `bitsandbytes` may not support the current OS, Python version, or hardware.
- CUDA-oriented quantization paths may fail on CPU-only machines.
- Some models may not support a chosen quantization configuration.
- Dynamic-int8 can increase load time.
- Aggressive quantization can harm output quality.
- Small prompt samples cannot prove broad quality behavior.

## Implementation Notes

- Main file: `src/airllm_ex05/runners/quantized_runner.py`
- Shared generation helper: `src/airllm_ex05/runners/baseline_runner.py`
- Metrics: `src/airllm_ex05/metrics.py`
- Tests: `tests/test_runners.py`

## Report Requirements

The report must separate three claims:

- Quantized generation was faster in the final measured run.
- Dynamic-int8 was the practical Windows CPU backend.
- The experiment does not prove that every memory or quality metric improves under
  quantization.
