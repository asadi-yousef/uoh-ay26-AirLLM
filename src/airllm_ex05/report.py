"""Analysis and Markdown report generation."""

import json
from pathlib import Path

from airllm_ex05.benchmark import load_results, save_results_csv
from airllm_ex05.config import ExperimentConfig
from airllm_ex05.cost_analysis import break_even_request_count, build_cost_curve
from airllm_ex05.plotting import plot_cost_curve, plot_metric


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
        plot_metric(results, "total_latency_seconds", config.outputs.figures_dir / "latency.png"),
        plot_metric(results, "tokens_per_second", config.outputs.figures_dir / "throughput.png"),
        plot_metric(results, "peak_ram_mb", config.outputs.figures_dir / "memory.png"),
        plot_cost_curve(cost_points, config.outputs.figures_dir / "cost_curve.png"),
    ]


def _report_body(config: ExperimentConfig, analysis: dict, results: list) -> str:
    failures = [result for result in results if result.status == "failed"]
    return f"""# Exercise 05 Technical Report

## Executive Summary

This report documents a reproducible local LLM experiment for baseline Hugging Face execution, AirLLM-style execution, and quantized execution. The project treats failed runs as valid measurements when the error is saved with configuration and hardware context.

## Hardware Specification

Run `uv run airllm-ex05 hardware` to refresh `results/raw/hardware.json`. The report should cite CPU model, core count, RAM, GPU, VRAM, CUDA availability, OS, Python version, and storage from that file.

## Selected Model

Configured model: `{config.model.name}`. The default model is intentionally small for pipeline verification. For the final submission experiment, replace it with a model large enough to stress local RAM/VRAM and justify the choice by parameter count, model format, expected memory footprint, and local hardware limits.

## Measurements

- Raw result count: {analysis["result_count"]}
- Successful runs: {analysis["success_count"]}
- Failed runs: {analysis["failure_count"]}
- Break-even monthly request volume: {analysis["break_even_monthly_requests"]}

Comparison tables are generated under `results/processed/`; plots are generated under `results/figures/`.

## Baseline Direct Run

The baseline runner attempts direct `transformers` loading. If loading fails because dependencies, RAM, VRAM, CUDA, or model compatibility are insufficient, the failure is saved as structured JSON and should be discussed as the baseline bottleneck.

## AirLLM Run

AirLLM is analyzed as a paging strategy: model layers are moved through memory instead of loading all weights at once. This can reduce peak memory pressure, but it usually increases latency because disk I/O becomes part of each forward pass.

## Quantization Run

The quantized runner attempts 4-bit or 8-bit loading through `transformers` quantization APIs. The report should compare memory, speed, and output quality against baseline and identify where compression crosses the quality red line.

## Prefill, Decode, TTFT, and TPOT

Prefill processes the prompt and is typically compute-bound. Decode produces one token at a time and is often memory-bandwidth-bound. TTFT approximates the prefill/user-visible startup delay, while TPOT captures steady-state decode cost.

## Economic Analysis

The cost model compares API token pricing with amortized local hardware, electricity, and maintenance. Prompt caching is represented by a cached-input discount assumption. Local inference becomes attractive at high volume, for privacy, or when data cannot leave the machine; APIs are better for low volume, bursty workloads, and operational simplicity.

## Negative Results and Limitations

Negative results are expected when optional packages are missing or hardware is too small. Current failed runs: {len(failures)}. Each failure should be interpreted through dependency availability, model size, memory pressure, and CPU/GPU constraints.

## Final Engineering Conclusions

The project provides the measurement and reporting pipeline required for the exercise. Final conclusions must be updated after real local runs with the selected stress model.
"""
