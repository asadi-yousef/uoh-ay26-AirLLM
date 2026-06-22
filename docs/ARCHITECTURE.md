# Architecture

## Purpose

This project is structured as a reproducible experiment harness rather than a one-off script.
The architecture separates configuration, execution, measurement, persistence, analysis, and
documentation so each part can be tested independently.

## Data Flow

```text
configs/experiment.yaml
        |
        v
airllm_ex05.config.load_config()
        |
        v
CLI command
        |
        +--> clears old runner-specific raw outputs
        |
        +--> hardware.py --------------------> results/raw/hardware.json
        |
        +--> baseline_runner.py -------------+
        +--> airllm_runner.py                +--> results/raw/*_p*_r*.json
        +--> quantized_runner.py ------------+--> results/raw/*_results.csv
        |
        v
report.analyze_results()
        |
        +--> results/processed/comparison_table.csv
        +--> results/processed/analysis.json
        +--> results/figures/*.png
        |
        v
README.md / docs/REPORT.md
```

## Components

| Component | Responsibility |
| --- | --- |
| `cli.py` | User-facing command dispatch and stale runner-output cleanup |
| `config.py` | Typed config loading and repository-relative path resolution |
| `models.py` | Pydantic schemas for hardware, metrics, and benchmark results |
| `hardware.py` | OS, Python, CPU, RAM, GPU, VRAM, CUDA, and storage collection |
| `metrics.py` | Token estimates, throughput, TPOT, RAM sampling, CUDA memory helpers |
| `benchmark.py` | JSON/CSV benchmark persistence |
| `runners/common.py` | Shared prompt iteration and structured failure creation |
| `runners/baseline_runner.py` | Direct Transformers inference |
| `runners/airllm_runner.py` | AirLLM load/generation attempt and shard path metadata |
| `runners/quantized_runner.py` | `bitsandbytes`, CPU dynamic-int8, and dynamic-int8 memory guard |
| `cost_analysis.py` | API/local cost estimates and break-even scan |
| `plotting.py` | Matplotlib figures |
| `report.py` | Analysis and report generation |

## Design Decisions

- Heavy model packages are lazy-imported so normal tests do not require model downloads.
- Every runner returns structured results instead of crashing the whole benchmark.
- Failed rows are preserved because negative results are valid assignment evidence.
- Runner commands remove stale JSON/CSV outputs for that runner before writing fresh results.
- Config owns paths, model choice, prompts, and cost assumptions.
- Raw local artifacts are ignored by Git; processed tables and figures are committed.
- Tests use fake dependencies for model paths so CI remains lightweight.
- The dynamic-int8 path checks cached checkpoint footprint against RAM before conversion.

## Final Evidence Shape

- Baseline has two successful 7B rows with latency, TTFT, TPOT, throughput, RAM, and VRAM.
- AirLLM has two failed 7B rows with `AttributeError`.
- Quantized has two failed 7B rows with load-stage `MemoryError`.
- Analysis therefore contains six rows, two successes, and four failures.
- Processed evidence is committed under `results/processed/`; raw JSON/CSV evidence remains
  local unless an evaluator explicitly requires it.

## Extension Points

- Add a new runner by implementing a `run_<name>(config)` function returning
  `list[BenchmarkResult]`.
- Add a metric by extending `MetricSummary`, updating CSV flattening in `benchmark.py`, and
  adding tests.
- Add a cost scenario by extending `CostConfig` and `build_cost_curve()`.
- Add a plot by extending `report._create_figures()`.
