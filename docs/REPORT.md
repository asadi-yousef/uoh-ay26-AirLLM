# Exercise 05 Technical Report

## Executive Summary

This report documents a reproducible local LLM experiment for baseline Hugging Face execution, AirLLM-style execution, and quantized execution. The project treats failed runs as valid measurements when the error is saved with configuration and hardware context.

## Hardware Specification

Run `uv run airllm-ex05 hardware` to refresh `results/raw/hardware.json`. The report should cite CPU model, core count, RAM, GPU, VRAM, CUDA availability, OS, Python version, and storage from that file.

## Selected Model

Configured model: `Qwen/Qwen2.5-3B-Instruct`. The default model is intentionally small for pipeline verification. For the final submission experiment, replace it with a model large enough to stress local RAM/VRAM and justify the choice by parameter count, model format, expected memory footprint, and local hardware limits.

## Measurements

- Raw result count: 6
- Successful runs: 4
- Failed runs: 2
- Break-even monthly request volume: None

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

Negative results are expected when optional packages are missing or hardware is too small. Current failed runs: 2. Each failure should be interpreted through dependency availability, model size, memory pressure, and CPU/GPU constraints.

## Final Engineering Conclusions

The project provides the measurement and reporting pipeline required for the exercise. Final conclusions must be updated after real local runs with the selected stress model.
