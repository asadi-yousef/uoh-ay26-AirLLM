# AirLLM Pipeline PRD

## Purpose

Provide an AirLLM execution path for the same model and prompts used by the baseline runner,
while preserving load and generation failures as structured experiment data.

## Context

The lecture presents AirLLM as a practical application of virtual memory ideas to LLM
inference. Instead of loading all model weights into RAM or VRAM, AirLLM moves through model
layers progressively, often using disk-backed shards. This can make a large model runnable on
modest hardware but usually increases latency because disk I/O enters the critical path.

## Relation to Exercise 05

Exercise 05 requires comparing direct local inference with AirLLM and explaining the
resource-allocation change. The report must connect AirLLM behavior to paging, virtual memory,
`mmap`, RAM/VRAM pressure, and latency/throughput tradeoffs.

## Functional Requirements

- Read model name and AirLLM layer-shard path from `configs/experiment.yaml`.
- Import `airllm` lazily so tests and non-AirLLM commands do not require it.
- Prefer `airllm.AutoModel` when available, with `AirLLMLlama2` as a fallback.
- Use the configured `layer_shards_saving_path`.
- Run the same prompt/run grid as other runners.
- Return `BenchmarkResult` objects for both successes and failures.
- Include AirLLM metadata such as layer-shard path or load-stage failure.

## Non-Functional Requirements

- AirLLM absence or incompatibility must not crash the whole CLI.
- The runner must be testable with fake modules.
- AirLLM caches and shards must remain ignored by Git.
- The report must clearly distinguish dependency failure, shard-creation success, and measured
  AirLLM generation performance.

## Inputs and Outputs

Inputs:

- `ExperimentConfig.model.name`
- `ExperimentConfig.airllm.layer_shards_saving_path`
- Prompt list, run count, and max new tokens
- Installed `airllm` package and its transitive dependencies

Outputs:

- `results/raw/airllm_p*_r*.json`
- `results/raw/airllm_results.csv`
- Processed comparisons and plots when metrics exist

## Acceptance Criteria

- `uv run airllm-ex05 airllm --config configs/experiment.yaml` writes one result per
  prompt/run.
- Missing `airllm`, missing transitive dependencies, or internal AirLLM load failures are saved
  as failed results with `metadata.stage = "load"`.
- Successful runs include generated text, latency, derived TPOT, throughput, peak RAM, and
  peak VRAM when CUDA exists.

## Risks and Failure Modes

- Version compatibility between AirLLM, Optimum, and Transformers.
- AirLLM API changes, for example no compatible `AutoModel` or `AirLLMLlama2`.
- Model architecture unsupported by installed AirLLM.
- Layer shard creation fills disk or uses a slow drive.
- CPU-only execution becomes too slow to complete within a practical timeout.

## Testing Strategy

Unit tests monkeypatch `importlib.import_module` with fake AirLLM classes. Tests verify both
missing-dependency failure behavior and a successful fake generation path without downloading
models.

## Implementation Notes Connected to Code

- Implemented in `src/airllm_ex05/runners/airllm_runner.py`.
- Shared success/failure measurement is in `runners/common.py`.
- Result constants are defined in `constants.py`.
- Current final run imports AirLLM successfully and creates Qwen2.5-3B layer shards, but fails
  during AirLLM's internal load path with `IndexError: list index out of range`.
