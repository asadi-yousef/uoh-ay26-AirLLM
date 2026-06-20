# Product Requirements Document

## Context

Exercise 05 requires a complete local-LLM engineering experiment, not only a working inference demo. The project must show how a large Hugging Face model behaves on local hardware when run directly, through AirLLM-style layer paging, and with quantization. Failures are valid experimental results when they are measured, saved, and analyzed.

## Goals

- Collect reproducible hardware information: CPU, cores, RAM, GPU, VRAM, CUDA, OS, Python, and storage.
- Run identical prompts through baseline, AirLLM, and quantized execution paths.
- Measure latency, TTFT, TPOT, throughput, memory, load time, status, output sample, and error details.
- Generate comparison tables, plots, and an economic analysis of local inference versus API usage.
- Produce a deep technical report connecting results to Prefill, Decode, VRAM/RAM pressure, paging, quantization, and on-premises economics.

## Non-Goals

- Do not require tests to download or run large models.
- Do not hide negative results or hardware failures.
- Do not store model weights, secrets, caches, PDFs, or heavy generated results in Git.

## Users

- Course evaluator reviewing reproducibility and engineering quality.
- Student running experiments on a personal machine with limited CPU/GPU/RAM.
- Future maintainer extending the benchmark to other models or quantization methods.

## Success Criteria

- All CLI commands run and save structured outputs or structured failures.
- The project uses `uv`, `pyproject.toml`, Ruff, pytest, and config-driven parameters.
- Core logic has lightweight unit tests with at least 85% coverage.
- `docs/REPORT.md` and `README.md` explain measurements, graphs, negative results, and conclusions.

## Constraints

- Heavy dependencies such as `transformers`, `airllm`, and `bitsandbytes` are optional at runtime and handled gracefully when missing.
- The default model is configurable and conservative enough to test the pipeline, while documentation explains how to choose a larger stress model.
- Python 3.11 or 3.12 is preferred because the lecture warns against relying on the newest Python for ML packages.
