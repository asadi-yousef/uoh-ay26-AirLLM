# Benchmarking PRD

## Objective

Provide a repeatable benchmark harness that records both successful inference and failed attempts. The assignment explicitly values analyzed negative results, so failures must contain enough context to explain the bottleneck.

## Required Metrics

- Model loading time.
- Total latency.
- Time to first token when observable.
- Time per output token.
- Tokens per second.
- Output token count and approximate input token count.
- Peak RAM and peak VRAM where practical.
- Success or failure status.
- Generated text sample or error message.

## Design

Each runner returns a common `BenchmarkResult` object. Serialization writes JSON for fidelity and CSV for tables. A memory sampler records process RSS and CUDA memory when available. If streaming token callbacks are unavailable, TTFT remains `null` and TPOT is derived from total latency and output tokens.

## Acceptance Criteria

- Same prompt list and run count are used across modes.
- Missing GPU/CUDA is recorded, not treated as a project failure.
- Exceptions are caught per experiment and saved with runner name, model name, prompt index, and hardware snapshot path.
