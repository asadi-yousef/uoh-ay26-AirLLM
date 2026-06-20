# uoh-ay26-AirLLM

Professional submission project for Exercise 05 in the AI Agents Orchestration course:
**Running a Massive LLM Locally: AirLLM, Quantization and Performance Benchmarking**.

The goal is not only to make a model produce text. The project provides a reproducible
engineering pipeline for documenting hardware, running baseline/AirLLM/quantized local
LLM experiments, saving failures as data, generating plots, comparing costs, and preparing
a technical report.

## Project Structure

- `configs/experiment.yaml` controls model choice, prompts, run counts, output paths,
  benchmark settings, AirLLM cache paths, quantization settings, and cost assumptions.
- `src/airllm_ex05/` contains the package, CLI, hardware collector, benchmark schemas,
  runners, analysis, plotting, and report generation.
- `docs/` contains PRDs, implementation plan, TODO, and `REPORT.md`.
- `results/` is the local output area. Raw model outputs and caches are ignored by Git;
  the small final figures are committed because the assignment requires visual evidence.
- `tests/` contains lightweight pytest tests that do not download models.

## Installation

Install `uv`, then create the environment:

```bash
uv sync
```

For real model execution, install optional model dependencies:

```bash
uv sync --extra models
```

Python 3.11 or 3.12 is recommended for the broadest ML-package compatibility. The project
allows Python 3.13 for tooling/tests, but some model packages may lag behind it.

## CLI Commands

```bash
uv run airllm-ex05 hardware
uv run airllm-ex05 baseline --config configs/experiment.yaml
uv run airllm-ex05 airllm --config configs/experiment.yaml
uv run airllm-ex05 quantized --config configs/experiment.yaml
uv run airllm-ex05 analyze --config configs/experiment.yaml
uv run airllm-ex05 report --config configs/experiment.yaml
```

The runner commands save one JSON file per prompt/run plus a CSV summary under
`results/raw/`. If a model cannot load because of missing packages, RAM/VRAM limits, CUDA
issues, or incompatibility, the failure is saved as a structured result.

## Configuration

The committed config is set for the final bounded stress experiment:

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Output length: `max_new_tokens: 16`
- Quantization: CPU `dynamic_int8`

For further experiments:

- Keep `max_new_tokens` low while validating the pipeline.
- Set `model.cache_dir` and `airllm.layer_shards_saving_path` to a disk with enough space.
- Adjust API pricing, hardware cost, electricity, and request volumes for the cost analysis.

Do not store Hugging Face tokens in code or config. Use environment variables or the normal
Hugging Face login flow.

## Results

- `results/raw/hardware.json`: CPU, cores, RAM, GPU, VRAM, CUDA, OS, Python, and storage.
- `results/raw/*.json`: raw per-run benchmark results.
- `results/raw/*_results.csv`: runner-level summaries.
- `results/processed/analysis.json`: success/failure and cost-analysis summary.
- `results/processed/comparison_table.csv`: flat comparison table.
- `results/figures/*.png`: latency, throughput, memory, and cost plots.
- `docs/REPORT.md`: generated Markdown report.

## Final Experiment Summary

The final local experiment uses `Qwen/Qwen2.5-3B-Instruct`, two prompts, one run per prompt,
and `max_new_tokens: 16` on a CPU-only Windows laptop.

- Hardware: Windows 11, Intel CPU, 4 physical cores, 8 logical cores, 15.70 GiB RAM, no
  CUDA-visible GPU or VRAM.
- Baseline: succeeded, but it was slow. Load time was 710.51 seconds; generation latencies were
  50.33 seconds and 62.87 seconds; average throughput was about 0.45 tokens/second.
- AirLLM: successfully created 76 layer shard files totaling about 6.17 GB, then failed during
  load with `IndexError: list index out of range`. This is reported as a negative compatibility
  result.
- Quantized: succeeded with CPU `torch.dynamic_int8`. Load time was 159.31 seconds; generation
  latencies were 11.88 seconds and 6.18 seconds; average throughput was about 3.40
  tokens/second.
- Analysis: 6 raw results, 4 successes, 2 failures, generated comparison tables and four plots.
  No local/API break-even appears in the configured request volumes.

| Runner | Load s | Avg latency s | Avg TTFT s | Avg TPOT s/token | Avg tok/s | Avg peak RAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 710.51 | 56.60 | 7.15 | 2.23 | 0.45 | 7739.84 |
| quantized dynamic-int8 | 159.31 | 9.03 | 3.00 | 0.33 | 3.40 | 9570.18 |

### Figures

![Latency comparison](results/figures/latency.png)

![Throughput comparison](results/figures/throughput.png)

![Peak RAM comparison](results/figures/memory.png)

![Cost curve](results/figures/cost_curve.png)

See `docs/REPORT.md` for the full technical analysis: Prefill versus Decode, TTFT/TPOT,
memory-bound behavior, AirLLM paging, quantization tradeoffs, output quality, and local versus
API economics.

## Quality Commands

```bash
uv run ruff check .
uv run pytest
```

The tests mock heavy model execution and verify config loading, metrics, hardware schema,
result serialization, cost analysis, report generation, runner failure handling, and CLI smoke
paths.

## Troubleshooting

- **`transformers` missing**: install optional dependencies with `uv sync --extra models`.
- **AirLLM unavailable or incompatible**: the AirLLM command records a failed result; discuss
  it in the report as a negative result.
- **No CUDA/GPU**: hardware collection records this and runners fall back or fail gracefully.
- **Out of memory**: reduce `max_new_tokens`, choose a smaller validation model, then run the
  intended stress model once the pipeline is verified.
- **Disk pressure**: AirLLM shards and Hugging Face caches can be large. Keep cache directories
  outside system-critical locations when needed.
- **Windows quantization**: `bitsandbytes` support may be limited. This project defaults to
  CPU `torch.dynamic_int8` for the observed Windows CPU-only hardware.

## Expected Hardware Limitations

The assignment expects a model that is uncomfortable for the local machine. Direct baseline
loading may fail, swap heavily, or run slowly. AirLLM may reduce memory pressure but usually
pays with high latency due to disk I/O and layer paging. Quantization should reduce memory and
bandwidth requirements, but aggressive compression can hurt output quality.

## Report Workflow

1. Run `hardware`.
2. Run `baseline`, `airllm`, and `quantized`.
3. Run `analyze`.
4. Run `report`.
5. Edit `docs/REPORT.md` only if you need to add human qualitative observations, screenshots,
   or final conclusions that cannot be inferred from raw measurements.

Negative results should remain in the report and be explained through Prefill/Decode,
TTFT/TPOT, RAM/VRAM pressure, CPU/GPU constraints, paging, and package compatibility.

## License and Credits

Course assignment and lecture materials are credited to Dr. Yoram Segal. This repository
contains student implementation code for the Exercise 05 submission. Model licenses depend on
the selected Hugging Face model and must be checked before use or sharing.
