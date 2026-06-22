# AirLLM Pipeline PRD

## Purpose

Provide an AirLLM execution path for the same model and prompts used by the baseline runner,
while preserving dependency, load, shard, and generation failures as structured experiment
data.

## Context

AirLLM is relevant because the lecture presents it as a practical application of virtual
memory ideas to LLM inference. Instead of requiring all model weights to be resident in RAM or
VRAM at once, AirLLM can shard and page layers through memory. This can make larger models more
reachable on constrained machines, but disk I/O and framework overhead can dominate latency.

## User Story

As the student, I want to run the same final model through AirLLM so I can compare direct
inference with a paging-style execution strategy and explain whether AirLLM helped, failed, or
introduced different bottlenecks on my hardware.

## Functional Requirements

- Read the model name from `ExperimentConfig.model.name`.
- Read the shard path from `ExperimentConfig.airllm.layer_shards_saving_path`.
- Import AirLLM lazily.
- Prefer `airllm.AutoModel` when available.
- Fall back to `AirLLMLlama2` when needed.
- Use the same prompt/run grid as the other runners.
- Save one result per prompt/run.
- Save load-stage failures for all prompts if model construction fails.
- Save generation-stage failures for the affected prompt/run if generation fails.
- Include runner name `airllm`.
- Include model name.
- Include prompt and run indices.
- Include error type and message on failure.
- Include AirLLM metadata such as shard path and failure stage.
- Keep AirLLM shards out of Git.

## Non-Functional Requirements

- Missing AirLLM must not crash unrelated commands.
- AirLLM tests must not require downloading a model.
- AirLLM tests must use fake modules/classes.
- The CLI must still write results when AirLLM fails.
- The report must distinguish shard creation from successful generation.
- Documentation must explain AirLLM through paging, `mmap`, disk I/O, RAM pressure, and latency.

## Acceptance Criteria

- `uv run airllm-ex05 airllm --config configs/experiment.yaml` writes result files.
- A missing `airllm` package is saved as structured failed results.
- An internal AirLLM load failure is saved as structured failed results.
- A successful fake AirLLM run is covered by tests.
- Final report includes the observed AirLLM failure rather than omitting the runner.

## Final Observed Evidence

- AirLLM imported successfully after dependency work.
- AirLLM imported successfully for the configured 7B model.
- AirLLM then failed before generation with
  `AttributeError: 'str' object has no attribute 'shape'`.
- Both final AirLLM prompt results are failed rows.
- No AirLLM generation latency, TTFT, TPOT, throughput, or output quality can be claimed.

## Risks

- AirLLM APIs can change.
- AirLLM may not support the selected architecture.
- AirLLM may depend on specific Transformers or Optimum versions.
- Shard creation may consume significant disk space.
- Slow disks can make paging impractical.
- CPU execution or a small laptop GPU can make even successful AirLLM generation too slow.

## Implementation Notes

- Main file: `src/airllm_ex05/runners/airllm_runner.py`
- Shared helpers: `src/airllm_ex05/runners/common.py`
- Result models: `src/airllm_ex05/models.py`
- Persistence: `src/airllm_ex05/benchmark.py`
- Tests: `tests/test_runners.py`

## Report Requirements

The report must state that AirLLM was attempted and generation did not happen. It must not
compare AirLLM speed against baseline because there are no successful AirLLM generation metrics
in the final evidence.
