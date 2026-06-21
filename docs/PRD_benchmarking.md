# Benchmarking PRD

## Purpose

Define the benchmark harness, metrics, data model, and serialization strategy for comparing
baseline, AirLLM, and quantized local inference attempts.

## Context

The assignment requires measured evidence. It also accepts negative results when they are
captured and analyzed. The benchmark therefore needs to produce useful data even when a runner
fails because of dependencies, memory limits, CUDA availability, package incompatibility, or
model architecture assumptions.

## Functional Requirements

- Iterate all configured prompts.
- Iterate all configured run indices.
- Use the same prompt/run grid for every runner.
- Measure model load time when loading succeeds.
- Measure total generation latency.
- Measure time to first token when streaming is available.
- Estimate input tokens.
- Estimate output tokens.
- Compute TPOT.
- Compute tokens per second.
- Sample process RAM.
- Record CUDA peak VRAM when CUDA exists.
- Save generated sample text.
- Save failure type and message.
- Save runner, model, prompt index, and run index for every result.
- Save one JSON file per result.
- Save one CSV summary per runner.
- Load mixed success/failure results for analysis.
- Exclude hardware JSON from benchmark-result loading.

## Non-Functional Requirements

- Tests must run offline.
- Tests must not download models.
- Heavy dependencies must be mocked.
- JSON must be readable and stable.
- CSV must be spreadsheet-friendly.
- Failure data must be as visible as success data.
- Raw outputs and heavy artifacts must remain outside the intended final commit.

## Result Schema

Each result stores:

- `runner`
- `model_name`
- `prompt_index`
- `run_index`
- `status`
- `metrics`
- `generated_text`
- `error_type`
- `error_message`
- `metadata`

Metrics store:

- `input_tokens`
- `output_tokens`
- `load_time_seconds`
- `total_latency_seconds`
- `time_to_first_token_seconds`
- `time_per_output_token_seconds`
- `tokens_per_second`
- `peak_ram_mb`
- `peak_vram_mb`

## Acceptance Criteria

- Every invoked runner produces one result per configured prompt/run.
- Load failures produce failed rows for every configured prompt/run.
- Generation failures preserve prompt/run context.
- Successful rows include latency and token-derived metrics.
- Analysis can process failed and successful rows together.
- The final comparison table includes AirLLM failed rows.

## Final Evidence

Current processed evidence contains:

- 6 raw benchmark rows.
- 4 successful rows.
- 2 failed rows.
- 2 baseline successes.
- 2 quantized successes.
- 2 AirLLM failures.

## Risks

- Approximate token counting can differ from true tokenizer counts.
- RAM sampling can miss short peaks.
- TTFT is only available for streaming paths.
- AirLLM metrics are unavailable if AirLLM fails before generation.
- CPU-only machines cannot provide real CUDA/VRAM measurements; this final run did have a
  CUDA-visible 4.0 GiB laptop GPU, while the dynamic-int8 quantized path ran on CPU.
- Very slow local generation can make large repeated experiments impractical.

## Implementation Notes

- Models: `src/airllm_ex05/models.py`
- Persistence: `src/airllm_ex05/benchmark.py`
- Metrics: `src/airllm_ex05/metrics.py`
- Runner shared code: `src/airllm_ex05/runners/common.py`
- Analysis/report: `src/airllm_ex05/report.py`
- Tests: `tests/test_benchmark.py`, `tests/test_metrics.py`, `tests/test_report.py`
