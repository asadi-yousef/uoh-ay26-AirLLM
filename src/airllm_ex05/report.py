"""Analysis and Markdown report generation."""

import json
from pathlib import Path

from airllm_ex05.benchmark import load_results, save_results_csv
from airllm_ex05.config import ExperimentConfig
from airllm_ex05.cost_analysis import break_even_request_count, build_cost_curve
from airllm_ex05.plotting import plot_cost_curve, plot_metric, plot_runner_outcomes


def analyze_results(config: ExperimentConfig) -> Path:
    """Analyze raw benchmark results and create processed outputs."""
    results = load_results(config.outputs.raw_dir)
    config.outputs.processed_dir.mkdir(parents=True, exist_ok=True)
    save_results_csv(results, config.outputs.processed_dir / "comparison_table.csv")
    avg_input, avg_output, seconds = _analysis_defaults(results)
    cost_points = build_cost_curve(avg_input, avg_output, seconds, config.cost)
    figures = _create_figures(config, results, cost_points)
    analysis = {
        "result_count": len(results),
        "success_count": sum(result.status == "success" for result in results),
        "failure_count": sum(result.status == "failed" for result in results),
        "cost_points": [point.model_dump() for point in cost_points],
        "break_even_monthly_requests": break_even_request_count(cost_points),
        "figures": [str(path) for path in figures if path is not None],
    }
    output = config.outputs.processed_dir / "analysis.json"
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return output


def generate_report(config: ExperimentConfig) -> Path:
    """Generate the Markdown technical report."""
    analysis_path = analyze_results(config)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    results = load_results(config.outputs.raw_dir)
    body = _report_body(config, analysis, results)
    config.outputs.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.report_path.write_text(body, encoding="utf-8")
    return config.outputs.report_path


def _analysis_defaults(results: list) -> tuple[int, int, float]:
    input_values = [result.metrics.input_tokens for result in results if result.metrics.input_tokens]
    output_values = [result.metrics.output_tokens for result in results if result.metrics.output_tokens]
    latency_values = [
        result.metrics.total_latency_seconds for result in results if result.metrics.total_latency_seconds
    ]
    avg_input = round(sum(input_values) / len(input_values)) if input_values else 256
    avg_output = round(sum(output_values) / len(output_values)) if output_values else 128
    seconds = sum(latency_values) / len(latency_values) if latency_values else 30.0
    return avg_input, avg_output, seconds


def _create_figures(config: ExperimentConfig, results: list, cost_points: list) -> list[Path | None]:
    return [
        plot_runner_outcomes(results, config.outputs.figures_dir / "outcomes.png"),
        plot_metric(results, "total_latency_seconds", config.outputs.figures_dir / "latency.png"),
        plot_metric(results, "tokens_per_second", config.outputs.figures_dir / "throughput.png"),
        plot_metric(results, "peak_ram_mb", config.outputs.figures_dir / "memory.png"),
        plot_cost_curve(cost_points, config.outputs.figures_dir / "cost_curve.png"),
    ]


def _report_body(config: ExperimentConfig, analysis: dict, results: list) -> str:
    baseline = [result for result in results if result.runner == "baseline"]
    airllm = [result for result in results if result.runner == "airllm"]
    quantized = [result for result in results if result.runner == "quantized"]
    return f"""# Exercise 05 Technical Report

## Executive Summary

This report documents a reproducible local LLM experiment for direct Hugging Face execution,
AirLLM-style layer paging, and quantized execution. The current 7B evidence contains successful
generation rows for all three runners after fixing the AirLLM input path and switching the final
quantized run to bitsandbytes 8-bit loading with CPU offload.

## Hardware Specification

The final hardware snapshot is stored in `results/raw/hardware.json`. The observed machine is a
Windows 11 laptop with 4 physical CPU cores, 8 logical CPU cores, 15.70 GiB RAM, and an NVIDIA
GeForce RTX 3050 Laptop GPU with 4.0 GiB dedicated VRAM visible to CUDA.

## Measurements

- Model: `{config.model.name}`
- Raw result count: {analysis["result_count"]}
- Successful runs: {analysis["success_count"]}
- Failed runs: {analysis["failure_count"]}
- Break-even monthly request volume: {analysis["break_even_monthly_requests"]}
- Baseline rows: {len(baseline)}
- AirLLM rows: {len(airllm)}
- Quantized rows: {len(quantized)}

Comparison tables are generated under `results/processed/`; plots are generated under `results/figures/`.

## Baseline Direct Run

The baseline runner uses `AutoTokenizer` and `AutoModelForCausalLM` with `device: auto`. In the
current evidence both baseline prompts succeeded. Model load took about 16.64 seconds, prompt
latency was about 210.26 to 216.45 seconds, throughput was about 0.10 to 0.14 output tokens per
second, and the CUDA/offload runtime memory metric was about 4.0 GiB.

## AirLLM Run

AirLLM is analyzed as a paging strategy: model layers are stored as shards and moved through
memory instead of keeping all weights resident at once. The earlier failure happened because the
runner passed a raw prompt string into AirLLM generation; the fixed runner tokenizes the prompt,
moves tensors to the model device when available, and decodes generated token IDs. In the current
evidence both AirLLM prompts succeeded. Load time was about 4.31 seconds, prompt latency was about
443.91 to 446.69 seconds, and throughput was about 0.05 to 0.07 output tokens per second.

## Quantization Run

The quantized runner supports `bitsandbytes`-style low-bit loading where available and CPU
`torch.dynamic_int8` for smaller CPU validation models. The 7B Windows path now uses
bitsandbytes 8-bit loading with fp32 CPU offload because dynamic-int8 conversion was too large
for local RAM. In the current evidence both quantized prompts succeeded. Load time was about
54.82 seconds, prompt latency was about 74.73 to 112.09 seconds, throughput was about 0.20 to
0.40 output tokens per second, and the CUDA/offload runtime memory metric was about 8.32 GiB.
This does not mean the quantized model fit inside dedicated GPU memory: the physical RTX 3050
Laptop GPU has 4.0 GiB dedicated VRAM, and bitsandbytes with Accelerate CPU offload can move
model/state through host RAM and offload paths.

## Memory Notes

The report separates physical hardware capacity, measured host RAM usage, and the benchmark's
CUDA/offload memory metric. The CUDA/offload memory column is a runner-level memory indicator
gathered from the execution stack. For the bitsandbytes CPU-offload run, it should not be
interpreted as pure dedicated VRAM residency, because the physical GPU has only 4 GB dedicated
VRAM. Its exact composition is backend-dependent, so the safest reading is measured runtime memory
pressure across CUDA/offload execution rather than physical GPU residency.

## Prefill, Decode, TTFT, and TPOT

Prefill processes the prompt and builds the initial context. Decode produces one token at a time
and is often memory-bandwidth-bound. TTFT approximates user-visible startup delay, while TPOT
captures steady decode cost. These metrics are available only for successful generation rows.

## Economic Analysis

The cost model compares API token pricing with amortized local hardware, electricity, and
maintenance. In the configured request volumes there is no API/local break-even point. APIs are
cheaper for the small measured prompt volumes; local inference remains useful for privacy,
offline execution, learning, and control.

## Screenshot Evidence

Evaluator-facing screenshots are stored under `docs/screenshots/` and embedded in the README.
The captured evidence includes the GitHub README overview, benchmark figures, hardware JSON,
processed comparison table, local AirLLM shard-cache folder, and passing Ruff/pytest output.

- README overview: `screenshots/01-github-readme-top/`
- README figures: `screenshots/02-readme-figures/`
- Hardware JSON: `screenshots/03-hardware-json/1.png`
- Comparison table: `screenshots/04-comparison-table/1.png`
- AirLLM shard-cache proof: `screenshots/05-airllm-shards-folder/1.png`
- Ruff and pytest: `screenshots/06-ruff-pytest-pass/`

![Comparison table evidence](screenshots/04-comparison-table/1.png)

![AirLLM shard-cache evidence](screenshots/05-airllm-shards-folder/1.png)

![Quality checks evidence](screenshots/06-ruff-pytest-pass/4.png)

## Negative Results and Limitations

Current failed runs: {analysis["failure_count"]}. Token counts are approximate, RAM sampling can
miss short spikes, AirLLM does not expose TTFT through the same streaming interface, and
bitsandbytes offload depends on CUDA, Accelerate, and Transformers compatibility.

## Final Engineering Conclusions

The final 7B evidence is realistic for constrained local LLM work: direct Transformers inference
works but is slow, AirLLM succeeds but is the slowest generation path on this machine, and
bitsandbytes 8-bit quantization with CPU offload gives the fastest observed generation among the
three tested runners. The repository connects the measurements to Prefill, Decode, paging,
quantization, memory pressure, and API/local cost.
"""
