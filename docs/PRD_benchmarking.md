# Benchmarking PRD

## Purpose

Define the benchmark harness, data model, metrics, and serialization strategy for comparing
baseline, AirLLM, and quantized local inference attempts.

## Context

The assignment requires measured evidence rather than informal observations. It also accepts
negative results when they are captured and analyzed. Therefore the benchmark system must save
both successful generation and failures with enough context to explain dependency,
compatibility, memory, or hardware bottlenecks.

## Relation to Exercise 05

Exercise 05 requires latency, throughput, TTFT, TPOT/ITL, peak RAM, peak VRAM, load time, and
qualitative outputs. These metrics support the conceptual analysis of Prefill versus Decode,
compute-bound versus memory-bound execution, AirLLM paging, quantization, and local/API cost.

## Functional Requirements

- Iterate all prompts and run indices consistently across runners.
- Measure model load time where loading succeeds.
- Measure total generation latency.
- Derive output token count, TPOT, and tokens per second.
- Approximate input token count without requiring tokenizer dependencies.
- Sample process RSS during generation.
- Record CUDA peak allocated memory when torch/CUDA are available.
- Save generated text samples, capped to a manageable length.
- Save error type, message, runner, model, prompt index, and run index for failures.
- Write per-result JSON and per-runner CSV.

## Non-Functional Requirements

- Tests must run without network and without model downloads.
- JSON output must remain stable and human-inspectable.
- CSV output must be spreadsheet-friendly.
- Missing optional dependencies must become data, not process crashes.
- Raw generated outputs must remain ignored by Git; final small plots and the comparison table
  may be committed for submission evidence.

## Inputs and Outputs

Inputs:

- `ExperimentConfig`
- Prompt list
- Runner-specific generation callable
- Optional torch/CUDA availability

Outputs:

- `BenchmarkResult`
- `MetricSummary`
- Raw JSON files in `results/raw/`
- CSV summaries in `results/raw/` and `results/processed/`

## Acceptance Criteria

- Every configured prompt/run produces exactly one result per invoked runner.
- Failed loads produce failed results for every prompt/run.
- Successful runs include latency and token-derived metrics.
- `load_results()` ignores `hardware.json` and loads only benchmark result files.
- Analysis can process mixed success and failure results.

## Risks and Failure Modes

- TTFT is measured for Transformers baseline and quantized runs through streaming generation;
  AirLLM TTFT remains unavailable when AirLLM fails before generation.
- Simple whitespace token counts differ from tokenizer-true counts.
- RAM sampling interval may miss short-lived peaks.
- CUDA memory reports are unavailable on CPU-only machines.
- Very slow model generation may exceed practical experiment time even if no code timeout is
  enforced in the runner.

## Testing Strategy

Tests cover metric math, JSON/CSV round trips, empty result directories, CLI smoke behavior,
fake runner success paths, and fake/missing dependency failure paths.

## Implementation Notes Connected to Code

- `models.py` defines `MetricSummary` and `BenchmarkResult`.
- `benchmark.py` implements `save_result`, `load_results`, and `save_results_csv`.
- `metrics.py` implements simple token counting, TPOT, throughput, RAM sampling, and CUDA peak
  memory.
- `runners/common.py` implements `iter_prompts`, `run_prompt`, and `failed_result`.
- Current final evidence includes two successful baseline results, two successful quantized
  results with TTFT, and two structured AirLLM failures after layer-shard creation.
