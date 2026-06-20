# Implementation Plan

## Source Requirements Extracted From PDFs

The assignment requires selecting a model that is too large or uncomfortable for local
hardware, documenting why, attempting direct baseline execution, then comparing AirLLM and
quantization on the same prompts. Required measurements include TTFT, TPOT or ITL, throughput,
total latency, peak RAM/VRAM, load time, estimated power/cost, and qualitative output quality.

The lecture frames AirLLM as layer-by-layer execution inspired by virtual memory, paging, and
`mmap`: fewer weights are resident at one time, but disk I/O increases latency. It frames
quantization as a memory and bandwidth optimization whose risk is quality loss. It also
separates Prefill, which is often compute-bound, from Decode, which is often memory-bound.

The software guidelines require modular code, planning docs, PRDs, `README.md`, `uv`,
`pyproject.toml`, config-driven settings, no secrets, ignored heavy artifacts, Ruff, pytest
coverage, and meaningful Git history.

## Architecture Overview

The project is a small Python package with a CLI front end and separated infrastructure:

- CLI layer: parses commands and dispatches work.
- Config layer: loads YAML into typed Pydantic models and resolves paths.
- Data-contract layer: defines benchmark, metric, and hardware schemas.
- Runner layer: contains direct baseline, AirLLM, and quantized inference attempts.
- Measurement layer: samples process RAM, derives TPOT and throughput, and records CUDA peak
  memory when available.
- Analysis layer: loads raw results, writes CSV/JSON summaries, creates figures, and computes
  cost curves.
- Documentation layer: README, PRDs, plan, TODO, and the final technical report.

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `cli.py` | Console command routing for hardware, benchmark, analysis, and report commands |
| `config.py` | YAML parsing, validation, and root-relative path resolution |
| `models.py` | Pydantic models for metrics, benchmark results, and hardware info |
| `benchmark.py` | Raw JSON and CSV result serialization |
| `hardware.py` | CPU/RAM/GPU/CUDA/OS/Python/storage collection |
| `metrics.py` | Token counting approximation, throughput, TPOT, RAM sampler, CUDA peak memory |
| `runners/common.py` | Shared prompt iteration, success measurement, and structured failures |
| `runners/baseline_runner.py` | Direct `transformers` causal-LM load and generation |
| `runners/airllm_runner.py` | Lazy AirLLM import, layer-shard path use, AirLLM generation |
| `runners/quantized_runner.py` | Lazy `transformers`/torch import and 4-bit/8-bit loading attempt |
| `cost_analysis.py` | API/local cost estimates and break-even scan |
| `plotting.py` | Matplotlib bar charts and cost curve |
| `report.py` | Processed analysis and generated report body |
| `shared/` | Logging, path, and validation utilities |

## Data Flow

1. User edits `configs/experiment.yaml`.
2. `load_config()` validates settings and resolves output/cache paths.
3. `hardware` writes `results/raw/hardware.json`.
4. Each runner iterates the same prompt/run grid.
5. Each attempt returns `BenchmarkResult` with either metrics and sample text or failure info.
6. CLI writes per-run JSON and per-runner CSV files in `results/raw/`.
7. `analyze` loads raw JSON, writes `comparison_table.csv`, `analysis.json`, and figures.
8. `report` regenerates analysis and writes `docs/REPORT.md`.
9. Human review expands the report with interpretation and final conclusions.

## Experiment Workflow

```bash
uv sync
uv sync --extra models
uv run airllm-ex05 hardware
uv run airllm-ex05 baseline --config configs/experiment.yaml
uv run airllm-ex05 airllm --config configs/experiment.yaml
uv run airllm-ex05 quantized --config configs/experiment.yaml
uv run airllm-ex05 analyze --config configs/experiment.yaml
uv run airllm-ex05 report --config configs/experiment.yaml
uv run ruff check .
uv run pytest
```

The current run used `sshleifer/tiny-gpt2`, two prompts, one run each, and 32 max new tokens.
Baseline succeeded; AirLLM and quantized modes failed during load due to dependency/platform
issues.

## Result File Structure

- `results/raw/hardware.json`: hardware snapshot.
- `results/raw/baseline_p0_r0.json`: per-prompt baseline result.
- `results/raw/airllm_p0_r0.json`: per-prompt AirLLM result or failure.
- `results/raw/quantized_p0_r0.json`: per-prompt quantized result or failure.
- `results/raw/*_results.csv`: runner summaries.
- `results/processed/analysis.json`: counts, cost points, break-even, figure paths.
- `results/processed/comparison_table.csv`: flattened comparison table.
- `results/figures/latency.png`: total latency bars where values exist.
- `results/figures/throughput.png`: throughput bars where values exist.
- `results/figures/memory.png`: peak RAM bars where values exist.
- `results/figures/cost_curve.png`: API versus local monthly cost.

Generated result files are ignored by Git. Important evidence is summarized in
`docs/REPORT.md`.

## Implementation Phases

1. Project skeleton, package metadata, `.gitignore`, and importable package.
2. Planning documentation and PRDs.
3. Config loader and shared utilities.
4. Hardware collector, metric helpers, and result schemas.
5. Baseline, AirLLM, and quantized runners with structured failure capture.
6. CLI commands and smoke tests.
7. Analysis, plots, cost model, and report generation.
8. Manual experiment run, report expansion, README polish, quality checks, and final commit.

## Design Decisions and Alternatives

- Use a tiny default model for pipeline validation, then require manual replacement for the
  final stress experiment.
- Use structured failed results instead of allowing runner exceptions to abort the experiment.
- Lazy-import heavy dependencies so tests and docs can run without model packages.
- Store raw generated evidence locally but ignore it in Git.
- Use simple token counting for dependency-light metrics; exact tokenizer counts can be added
  later.
- Use non-streaming generation for portability; exact TTFT requires a future streaming hook.

## Mapping to Assignment Requirements

| Assignment requirement | Current project support | Current evidence |
| --- | --- | --- |
| Hardware documentation | `hardware.py` and `hardware` CLI | Present in ignored `hardware.json` |
| Model choice and justification | Config and report | Current model is validation-only |
| Baseline run | `baseline_runner.py` | Succeeded twice |
| AirLLM run | `airllm_runner.py` | Failed at load due to missing dependency |
| Quantization run | `quantized_runner.py` | Failed at load due to missing `bitsandbytes` |
| TTFT/TPOT/throughput/latency | Metrics schema and derived TPOT/throughput | TPOT/throughput present for baseline; TTFT missing |
| RAM/VRAM | RAM sampler and CUDA peak helper | RAM present for baseline; no VRAM detected |
| Plots/tables | `plotting.py`, `report.py` | Generated from current evidence |
| Economic analysis | `cost_analysis.py` | No break-even in configured volumes |
| Negative results | Structured failures | AirLLM and quantized failures preserved |
| Final technical report | `docs/REPORT.md` | Expanded with real validation results |
