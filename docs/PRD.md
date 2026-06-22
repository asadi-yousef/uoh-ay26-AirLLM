# Product Requirements Document

## Purpose

Build a reproducible Exercise 05 local-LLM experiment package that documents local hardware,
runs direct baseline inference, attempts AirLLM inference, attempts quantized inference,
analyzes the resulting evidence, generates plots, compares local and API economics, and
produces a technical report. The product is not just a one-off notebook. It is an engineering
submission that can be rerun from a config file and CLI.

The project must treat failed runs as first-class results. A missing backend, unsupported
model architecture, RAM/VRAM limitation, or AirLLM runtime failure is useful evidence when it
is stored with model, prompt, runner, and hardware context.

## Assignment Context

Exercise 05 asks for a local experiment with a large or uncomfortable LLM. The analysis must
explain the outcome through the course concepts of CPU/GPU execution, RAM, VRAM, Prefill,
Decode, TTFT, TPOT/ITL, AirLLM paging, quantization, and on-premises economics.

The software guidelines require a professional repository:

- `uv` and `pyproject.toml`
- Modular source code under `src/`
- Config-driven behavior
- Tests that do not require private services or large downloads
- Ruff formatting/linting
- Documentation and planning files
- Git hygiene
- No private PDFs, secrets, model weights, caches, or heavy generated artifacts in commits

## Product Scope

In scope:

- Hardware collection for the local machine.
- Baseline Hugging Face Transformers inference.
- AirLLM inference attempt with layer-shard path support.
- Quantized inference attempt with `bitsandbytes`, CPU dynamic-int8 support, and a memory guard
  for oversized local dynamic-int8 conversion.
- Structured success and failure records.
- JSON and CSV result serialization.
- Processed comparison table.
- Latency, throughput, memory, and cost plots.
- Cost comparison between local inference and API use.
- Generated Markdown report plus manual interpretation.
- README-level evaluator summary.
- Lightweight automated tests.

Out of scope:

- Training or fine-tuning a model.
- Building a web UI.
- Guaranteeing AirLLM compatibility for every model architecture.
- Guaranteeing exact tokenizer-level metrics for every backend.
- Committing downloaded model weights or AirLLM shards.
- Hiding failed experiment paths from the final analysis.

## Users

Primary user:

- The student running the final local experiment and submitting the repo.

Secondary users:

- Course evaluator reviewing the code, docs, and plots.
- Future student or developer who wants to rerun the harness with a different model.

## Functional Requirements

### Configuration

- Load all experiment settings from `configs/experiment.yaml`.
- Validate config with typed models.
- Resolve paths relative to the repository root.
- Configure model name, cache path, trust-remote-code flag, and device policy.
- Configure prompts, run count, warmup count, max new tokens, and timeout value.
- Configure raw, processed, figure, and report output paths.
- Configure AirLLM layer-shard path.
- Configure quantization mode, bit width, and compute dtype.
- Configure local/API cost assumptions.

### Hardware

- Collect OS and Python version.
- Collect CPU name where available.
- Collect physical and logical core counts.
- Collect total RAM.
- Detect CUDA availability.
- Collect GPU name and VRAM when CUDA is available.
- Record CPU-only state when no GPU is visible.
- Write `results/raw/hardware.json`.

### Benchmark Runners

- Run every configured prompt and run index for each invoked runner.
- Save one result JSON per prompt/run.
- Save one runner-level CSV summary.
- Measure load time when model loading succeeds.
- Measure total generation latency.
- Measure TTFT for streaming Transformers-based paths.
- Derive TPOT and throughput.
- Sample peak process RAM during generation.
- Record CUDA peak VRAM when available.
- Save generated output sample text.
- Save structured failures with error type and message.
- Continue the benchmark when one runner fails.

### Baseline

- Load tokenizer and model through Transformers.
- Use configured cache directory.
- Respect `trust_remote_code`.
- Use streaming generation where possible to measure TTFT.
- Produce success metrics or structured load/generation failures.

### AirLLM

- Import AirLLM lazily.
- Use configured layer-shard path.
- Prefer `airllm.AutoModel` when available.
- Fall back to `AirLLMLlama2` when needed.
- Preserve AirLLM dependency and runtime failures as results.
- Document shard creation separately from successful generation.

### Quantization

- Support `bitsandbytes` 4-bit and 8-bit configuration.
- Support CPU `torch.dynamic_int8` for Windows CPU validation.
- Preserve missing backend errors as structured failures.
- Include quantization method metadata in successful results.
- Compare memory, speed, and output quality against baseline only when quantized generation
  succeeds.
- Preserve dynamic-int8 RAM-pressure failures as structured load-stage evidence.

### Analysis And Report

- Load all benchmark result JSON files.
- Ignore `hardware.json` when loading benchmark results.
- Write `results/processed/comparison_table.csv`.
- Write `results/processed/analysis.json`.
- Generate latency, throughput, memory, and cost plots.
- Generate `docs/REPORT.md`.
- Preserve failed rows in processed evidence.
- Make README summarize the final result and link to the full report.

## Non-Functional Requirements

- Tests must run without downloading models.
- Heavy ML dependencies must be optional or lazy-imported.
- CLI commands must be reproducible with `uv run`.
- Result files must be human-inspectable.
- The code must be modular and easy to review.
- Ruff and pytest must pass before submission.
- The final repo must avoid private or oversized artifacts.
- Documentation must be explicit about limitations and negative results.

## Final Configured Experiment

The final bounded stress experiment is configured as:

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Prompts: two conceptual prompts about Prefill/Decode and paging
- Runs: one per prompt
- Output length: 16 max new tokens
- Hardware: Windows 11 laptop with 15.70 GiB RAM and an RTX 3050 Laptop GPU visible to CUDA
- Quantization: bitsandbytes 8-bit with fp32 CPU offload
- AirLLM cache: `airllm_cache/qwen2_5_7b/layer_shards`

This model was selected because it is large enough to be slow and uncomfortable on the target
machine while remaining bounded enough to complete a short experiment.

## Inputs

- `configs/experiment.yaml`
- Local Python environment
- Optional model dependencies from `uv sync --extra models`
- Hugging Face cache
- AirLLM layer-shard directory
- Prompt list
- Cost assumptions

## Outputs

- `results/raw/hardware.json`
- `results/raw/baseline_p*_r*.json`
- `results/raw/airllm_p*_r*.json`
- `results/raw/quantized_p*_r*.json`
- `results/raw/*_results.csv`
- `results/processed/analysis.json`
- `results/processed/comparison_table.csv`
- `results/figures/latency.png`
- `results/figures/throughput.png`
- `results/figures/memory.png`
- `results/figures/outcomes.png`
- `results/figures/cost_curve.png`
- `docs/REPORT.md`

## Acceptance Criteria

- `uv run airllm-ex05 hardware` writes hardware JSON.
- `uv run airllm-ex05 baseline --config configs/experiment.yaml` writes one result per
  prompt/run.
- `uv run airllm-ex05 airllm --config configs/experiment.yaml` writes one result per
  prompt/run, even when AirLLM fails.
- `uv run airllm-ex05 quantized --config configs/experiment.yaml` writes one result per
  prompt/run.
- `uv run airllm-ex05 analyze --config configs/experiment.yaml` writes processed outputs and
  plots where data exists.
- `uv run airllm-ex05 report --config configs/experiment.yaml` writes the report.
- Failed runs are not silently dropped.
- README includes final metrics and figures.
- `uv run ruff check .` passes.
- `uv run pytest` passes.

## Final Evidence To Preserve

- Hardware: Windows 11, 4 physical cores, 8 logical cores, 15.70 GiB RAM, NVIDIA GeForce
  RTX 3050 Laptop GPU, 4.0 GiB VRAM, CUDA available.
- Baseline: succeeded on both final prompts but slowly.
- AirLLM: succeeded on both prompts after tokenized input handling and Transformers 4.45.x
  compatibility pinning.
- Quantized: succeeded on both prompts with bitsandbytes 8-bit loading and CPU offload.
- Analysis: 6 raw results, 6 successes, 0 failures.
- Cost model: no break-even within configured monthly request volumes.

## Risks

- AirLLM compatibility can vary by version and model architecture.
- CPU inference can be too slow for 7B-class models, and a 4 GiB laptop GPU is too small
  for comfortable unquantized large-model experimentation.
- `bitsandbytes` may not work on every Windows, CUDA, Python, or hardware combination.
- CPU dynamic-int8 can temporarily need more RAM than the compressed model footprint and can be
  infeasible for a 7B model on a 16 GiB RAM laptop.
- Python 3.12 was used for the final hardware snapshot; Python 3.13 may be ahead of some ML
  package support if the project is run elsewhere.
- Approximate token counts can differ from tokenizer-true counts.
- RAM sampling can miss short spikes.
- Cost assumptions are illustrative rather than measured from wall power.

## Testing Strategy

Tests use fake dependencies and temporary directories. They cover config loading, path
resolution, hardware data models, metrics, result serialization, cost calculations, report
generation, CLI smoke paths, and runner success/failure behavior with mocked heavy packages.

## Module Map

- `cli.py`: command routing.
- `config.py`: config models and path resolution.
- `models.py`: hardware, metric, and benchmark contracts.
- `benchmark.py`: JSON/CSV result persistence.
- `hardware.py`: local hardware snapshot.
- `metrics.py`: token estimates, throughput, TPOT, RAM, and VRAM helpers.
- `runners/common.py`: shared prompt iteration and structured failure handling.
- `runners/baseline_runner.py`: direct Transformers path.
- `runners/airllm_runner.py`: AirLLM path.
- `runners/quantized_runner.py`: quantized path.
- `tests/test_quantized_guard.py`: dynamic-int8 cache-size and memory-guard behavior.
- `cost_analysis.py`: API/local economics.
- `plotting.py`: figures.
- `report.py`: report generation.
- `shared/`: logging, path, and validation helpers.
