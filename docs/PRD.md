# Product Requirements Document

## Purpose

Build a reproducible Exercise 05 local-LLM experiment package that documents hardware, runs
direct baseline inference, attempts AirLLM inference, attempts quantized inference, analyzes
results, generates plots, and produces a technical report. The project must treat failures as
first-class results because the assignment explicitly values well-analyzed negative outcomes.

## Context

Exercise 05 asks students to run a large or uncomfortable LLM locally and explain what happens
through the lecture concepts of CPU/GPU execution, VRAM, RAM, Prefill, Decode, AirLLM paging,
quantization, and on-premises economics. The software guidelines require professional
structure: `uv`, `pyproject.toml`, modular source files, config files, tests, Ruff, Git hygiene,
and no private PDFs, secrets, model weights, caches, or heavy generated files in commits.

## Relation to Exercise 05

This project is the experiment harness and documentation system for the assignment. It does
not hardcode a final model; the model is selected in `configs/experiment.yaml` so the same code
can first validate the pipeline with a tiny model and then run the final stress experiment.

## Functional Requirements

- Load all experiment settings from YAML.
- Collect CPU, core count, RAM, GPU, VRAM, CUDA, OS, Python, and storage data.
- Run the same prompts and run counts through baseline, AirLLM, and quantized runners.
- Save one JSON result per prompt/run and one CSV summary per runner.
- Capture load-stage and generation-stage failures without crashing the whole benchmark.
- Generate a processed comparison table, analysis JSON, metric plots, cost plot, and report.
- Keep generated results, models, caches, private PDFs, and secrets out of Git.

## Non-Functional Requirements

- The code must run tests without downloading models.
- Heavy model dependencies must be optional or lazily imported.
- Paths must be config-driven and resolved relative to the project root.
- CLI commands must be simple and reproducible through `uv run`.
- Result files must be serializable, inspectable, and stable enough for report generation.
- Ruff and pytest must pass before submission.

## Inputs and Outputs

Inputs:

- `configs/experiment.yaml`
- Local Python environment and optional ML dependencies
- Hugging Face model cache and AirLLM cache paths
- Prompt list and benchmark settings
- Cost-analysis assumptions

Outputs:

- `results/raw/hardware.json`
- `results/raw/*_p*_r*.json`
- `results/raw/*_results.csv`
- `results/processed/analysis.json`
- `results/processed/comparison_table.csv`
- `results/figures/*.png`
- `docs/REPORT.md`

## Acceptance Criteria

- `uv run airllm-ex05 hardware` writes hardware JSON.
- `uv run airllm-ex05 baseline --config configs/experiment.yaml` writes success or failure
  records for every prompt/run.
- `uv run airllm-ex05 airllm --config configs/experiment.yaml` writes success or failure
  records for every prompt/run.
- `uv run airllm-ex05 quantized --config configs/experiment.yaml` writes success or failure
  records for every prompt/run.
- `uv run airllm-ex05 analyze --config configs/experiment.yaml` writes processed tables and
  plots where data exists.
- `uv run airllm-ex05 report --config configs/experiment.yaml` writes a report that can be
  manually expanded with qualitative analysis.
- `uv run ruff check .` and `uv run pytest` pass.

## Risks and Failure Modes

- Final model may be too small to satisfy the assignment's stress requirement.
- Large model download or AirLLM shard creation may require substantial disk space.
- CPU-only Windows may not support common 4-bit `bitsandbytes` paths.
- AirLLM may be incompatible with current package versions or model architecture.
- Python 3.13 may be newer than some ML packages support.
- Non-streaming generation leaves TTFT unmeasured.

## Testing Strategy

Tests use fakes and temporary directories. They cover config loading, path resolution,
hardware schema, metrics, result serialization, cost calculations, report generation, CLI
smoke paths, and runner success/failure behavior with mocked heavy dependencies.

## Implementation Notes

- `src/airllm_ex05/cli.py` owns the command surface.
- `config.py` defines Pydantic config models and path resolution.
- `models.py` defines benchmark and hardware data contracts.
- `benchmark.py` writes/loads raw result JSON and CSV summaries.
- `hardware.py` collects environment data with `psutil`, platform APIs, and optional torch.
- `runners/` contains baseline, AirLLM, and quantized execution paths.
- `report.py`, `plotting.py`, and `cost_analysis.py` create processed outputs and economics.
- `shared/` holds path, validation, and logging helpers.
