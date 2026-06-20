# Implementation Plan

## Source Requirements Extracted From PDFs

The assignment requires selecting a model too large or uncomfortable for the local hardware, documenting why, attempting a direct baseline run, then comparing AirLLM and quantization. Measurements must include TTFT, TPOT or ITL, throughput, total latency, peak RAM/VRAM, load time, estimated power or cost, and output quality observations. The report must analyze Prefill versus Decode, compute-bound versus memory-bound behavior, VRAM/RAM bottlenecks, virtual memory, paging, AirLLM resource allocation, quantization tradeoffs, and API versus on-premises economics.

The lecture frames AirLLM as layer-by-layer execution inspired by virtual memory paging and `mmap`: only the needed layer is resident while other weights remain on disk. This reduces memory pressure but increases latency because disk I/O becomes central. Quantization lowers precision to reduce memory and bandwidth needs, with a quality red line where compression damages output usefulness.

The software guidelines require planning documents, modular code, small files, `uv`, `pyproject.toml`, config files instead of hardcoded values, no secrets, Ruff with zero violations, tests with at least 85% coverage, clear errors, relative paths, and a Git history made of logical commits.

## Milestones

1. Project skeleton, package metadata, `.gitignore`, and importable package.
2. Planning documentation and PRDs.
3. Config loader and shared utilities.
4. Hardware collector, metric schemas, and serialization.
5. Baseline, AirLLM, and quantized runners with structured failure capture.
6. CLI commands and smoke tests.
7. Analysis, plots, cost model, and report generation.
8. README polish and final quality checks.

## Verification Strategy

- Unit tests use temporary directories and mocks, never model downloads.
- CLI smoke tests call cheap commands and validate output files.
- Ruff enforces imports, annotations, style, and common bug patterns.
- Generated heavy outputs are ignored; small docs and source files are committed.
