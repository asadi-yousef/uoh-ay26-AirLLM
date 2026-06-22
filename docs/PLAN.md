# Implementation Plan

## Source Requirements Extracted From Course Material

The assignment requires selecting a model that is too large or uncomfortable for the local
machine, documenting the hardware, attempting direct execution, then comparing AirLLM and
quantization on the same prompts. The required measurements include TTFT, TPOT or ITL,
throughput, total latency, peak RAM, peak VRAM where available, load time, estimated cost, and
qualitative output quality.

The lecture frames AirLLM as layer-by-layer execution inspired by virtual memory, paging, and
`mmap`. The benefit is lower resident weight pressure. The cost is extra disk I/O and
coordination overhead. The lecture frames quantization as a lower-precision representation that
can reduce memory and bandwidth requirements, especially during decode, while risking quality
loss.

The software guidelines require a normal software project: `uv`, `pyproject.toml`, modular
source files, config files, tests, Ruff, Git hygiene, PRDs, README, and no private or oversized
artifacts.

## Planned Architecture

The system is a Python package with a CLI front end and separated layers:

- CLI layer for user commands.
- Config layer for YAML parsing and path resolution.
- Data-contract layer for hardware, metrics, and benchmark result schemas.
- Runner layer for baseline, AirLLM, and quantized execution paths.
- Measurement layer for latency, TTFT, TPOT, throughput, RAM, and VRAM.
- Persistence layer for JSON and CSV results.
- Analysis layer for processed tables, summary JSON, figures, and cost model.
- Documentation layer for README, PRDs, TODO, plan, and technical report.

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `cli.py` | Parse commands and dispatch hardware, runner, analysis, and report flows |
| `config.py` | Define Pydantic config models, load YAML, resolve paths |
| `models.py` | Define `HardwareInfo`, `MetricSummary`, and `BenchmarkResult` |
| `benchmark.py` | Save/load result JSON and write CSV summaries |
| `hardware.py` | Collect OS, Python, CPU, RAM, CUDA, GPU, VRAM, and storage data |
| `metrics.py` | Token estimates, throughput, TPOT, RAM sampler, CUDA peak helper |
| `runners/common.py` | Prompt iteration, successful run measurement, failed result creation |
| `runners/baseline_runner.py` | Direct Transformers model loading and generation |
| `runners/airllm_runner.py` | AirLLM import, layer-shard path handling, generation attempt |
| `runners/quantized_runner.py` | `bitsandbytes` and CPU dynamic-int8 quantized paths |
| `cost_analysis.py` | Local/API monthly cost estimates and break-even scan |
| `plotting.py` | Latency, throughput, memory, and cost figures |
| `report.py` | Generated Markdown report body |
| `shared/logging_utils.py` | Logging setup |
| `shared/paths.py` | Root and directory helpers |
| `shared/validation.py` | Small validation helpers |

## Data Flow

1. User edits `configs/experiment.yaml`.
2. `load_config()` validates settings.
3. Config paths are resolved relative to the repository root.
4. `hardware` writes `results/raw/hardware.json`.
5. A runner command loads the config.
6. The runner loads or attempts to load its model path.
7. The runner iterates prompts and run indices.
8. Each attempt returns a `BenchmarkResult`.
9. The CLI writes per-run JSON files and a runner CSV.
10. `analyze` loads raw benchmark JSON files.
11. `analyze` writes `comparison_table.csv`, `analysis.json`, and figures.
12. `report` regenerates analysis and writes `docs/REPORT.md`.
13. Human review updates the final narrative and checks consistency.

## Implemented Workflow

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

## Implementation Phases

### Phase 1: Skeleton

- Create package structure.
- Create `pyproject.toml`.
- Configure dependencies and optional dependencies.
- Configure Ruff and pytest.
- Add result directories and `.gitkeep` files.
- Add `.gitignore` policy for caches, raw results, private PDFs, and bytecode.

### Phase 2: Planning Docs

- Write main PRD.
- Write implementation plan.
- Write specialized PRDs for benchmarking, AirLLM, and quantization.
- Write TODO ledger.
- Write initial README structure.

### Phase 3: Config And Contracts

- Implement Pydantic config models.
- Implement YAML loading.
- Implement root-relative path resolution.
- Implement hardware and benchmark result models.
- Add config and model tests.

### Phase 4: Measurement Infrastructure

- Implement approximate token counting.
- Implement throughput and TPOT helpers.
- Implement RAM sampler.
- Implement CUDA peak memory helper.
- Implement JSON/CSV persistence.
- Add tests for metric math and serialization.

### Phase 5: Runners

- Implement baseline Transformers runner.
- Implement shared runner helpers.
- Implement structured failure capture.
- Implement AirLLM runner with lazy imports and shard path.
- Implement quantized runner with `bitsandbytes` and dynamic-int8 support.
- Add fake dependency tests so CI does not download models.

### Phase 6: CLI

- Add `hardware`.
- Add `baseline`.
- Add `airllm`.
- Add `quantized`.
- Add `analyze`.
- Add `report`.
- Add CLI smoke tests.

### Phase 7: Analysis

- Implement result loading.
- Implement processed comparison table.
- Implement analysis summary JSON.
- Implement latency, throughput, memory, and cost plots.
- Implement cost model and break-even scan.
- Add analysis/report tests.

### Phase 8: Final Local Experiment

- Validate pipeline with a tiny model first.
- Select `Qwen/Qwen2.5-7B-Instruct` as the final stress model.
- Keep max output short to complete on constrained local hardware.
- Run baseline.
- Run AirLLM.
- Run quantized.
- Run analysis and report.
- Update README and docs with final evidence.
- Run Ruff and pytest.

## Design Decisions

- Use a CLI package rather than a notebook so the evaluator can rerun commands.
- Use config-driven paths so model/cache/output locations are not hard-coded.
- Save failures as result objects so negative outcomes remain analyzable.
- Lazy-import heavy dependencies so tests and docs can run without optional model packages.
- Use CPU dynamic-int8 as the Windows quantization attempt and preserve a structured
  RAM-pressure failure when the 7B conversion is too large.
- Add a memory guard before dynamic-int8 conversion so the benchmark writes failed rows instead
  of crashing after loading a large checkpoint.
- Keep `bitsandbytes` support for CUDA/non-Windows environments.
- Commit final plots and processed table because they are small and useful to reviewers.
- Keep raw model outputs, model caches, and AirLLM shards out of the intended commit.
- Use approximate token counts to keep the analysis dependency-light.
- Measure TTFT through streaming where Transformers supports it.

## Alternatives Considered

- Notebook-only implementation: rejected because it is harder to test and rerun cleanly.
- Committing raw JSON outputs: rejected as a default because raw outputs can be noisy and large,
  though local raw evidence is still generated.
- Using only a tiny model: rejected for final submission because the assignment expects stress.
- Using only `bitsandbytes`: rejected for final Windows validation because backend support was
  not reliable.
- Hiding AirLLM failure: rejected because the assignment values analyzed negative outcomes.

## Final Experiment Record

Final stress model: `Qwen/Qwen2.5-7B-Instruct`

Final prompts:

- Explain the difference between prefill and decode in local LLM inference.
- Summarize why paging can help run a model that does not fit in VRAM.

Final benchmark settings:

- Runs: 1
- Warmups: 0
- Max new tokens: 16
- Timeout setting: 900 seconds

Observed final result:

- Baseline succeeded for both prompts.
- AirLLM failed before generation with `AttributeError: 'str' object has no attribute 'shape'`.
- Quantized dynamic-int8 failed safely for both prompts with load-stage `MemoryError`.
- Analysis produced six result rows: two successes and four failures.
- No break-even appeared in the configured cost volumes.

## Mapping To Assignment Requirements

| Assignment requirement | Project support | Final evidence |
| --- | --- | --- |
| Hardware documentation | `hardware.py` and `hardware` CLI | Windows, 15.70 GiB RAM, RTX 3050 Laptop GPU |
| Model choice | Config, README, report | Qwen2.5-7B as the hardware-boundary stress model |
| Direct baseline | `baseline_runner.py` | Two successful runs |
| AirLLM | `airllm_runner.py` | Attempted 7B run, then failed with `AttributeError` |
| Quantization | `quantized_runner.py` | Two dynamic-int8 memory-guard failures |
| TTFT | Streaming Transformers generation | Present for baseline; unavailable for failed AirLLM/quantized rows |
| TPOT/throughput | `metrics.py` | Present for baseline; unavailable for failed AirLLM/quantized rows |
| RAM/VRAM | RAM sampler and CUDA helper | RAM present; CUDA-visible 4.0 GiB VRAM |
| Plots/tables | `plotting.py` and `report.py` | Four plots and comparison CSV |
| Cost analysis | `cost_analysis.py` | No configured break-even |
| Negative results | Structured failures | AirLLM and quantized failures preserved |
| Quality review | Report/README narrative | Baseline output review only for final evidence |

## Verification Plan

- Run `uv run ruff check .`.
- Run `uv run pytest`.
- Confirm comparison table matches report values.
- Confirm README plots render.
- Confirm ignored artifacts are not staged.
- Confirm private PDFs are not staged.
- Confirm AirLLM failure is described accurately.
- Confirm quantization claims match measured evidence.
