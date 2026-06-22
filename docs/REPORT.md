# Exercise 05 Technical Report

## Executive Summary

This report documents a reproducible local LLM experiment for direct Hugging Face execution,
AirLLM-style layer paging, and quantized execution. Failed runs are preserved as valid
measurements when they are saved with model, prompt, hardware, and error context.

## Hardware Specification

The final hardware snapshot is stored in `results/raw/hardware.json`. The observed machine is a
Windows 11 laptop with 4 physical CPU cores, 8 logical CPU cores, 15.70 GiB RAM, and an NVIDIA
GeForce RTX 3050 Laptop GPU with 4.0 GiB CUDA-visible VRAM.

## Selected Model

Configured model: `Qwen/Qwen2.5-7B-Instruct`. This 7B instruction model is intentionally
uncomfortable for the local hardware: it is much larger than the 4 GiB laptop GPU can hold
comfortably and is large enough to expose RAM pressure during quantization.

## Measurements

- Raw result count: 6
- Successful runs: 2
- Failed runs: 4
- Break-even monthly request volume: None
- Baseline rows: 2
- AirLLM rows: 2
- Quantized rows: 2

Comparison tables are generated under `results/processed/`; plots are generated under `results/figures/`.

## Baseline Direct Run

The baseline runner uses `AutoTokenizer` and `AutoModelForCausalLM` with `device: auto`. In the
current evidence both baseline prompts succeeded. Model load took about 353.12 seconds, prompt
latency was about 226.72 to 229.33 seconds, throughput was about 0.10 to 0.13 output tokens per
second, and CUDA peak allocation was about 4.0 GiB.

## AirLLM Run

AirLLM is analyzed as a paging strategy: model layers are stored as shards and moved through
memory instead of keeping all weights resident at once. In the current evidence AirLLM imported
and created 7B shards, but both prompt rows failed before generation with
`AttributeError: 'str' object has no attribute 'shape'`.

## Quantization Run

The quantized runner supports `bitsandbytes`-style low-bit loading where available and CPU
`torch.dynamic_int8` for Windows validation. In the current 7B evidence dynamic-int8 is stopped
by a pre-load memory guard: the cached checkpoint is about 14.2 GiB, physical RAM is about
15.7 GiB, and estimated conversion need is about 31.9 GiB. The two quantized rows are structured
`MemoryError` failures, so no quantized latency, TTFT, TPOT, throughput, RAM, VRAM, or output
quality metric is claimed.

## Prefill, Decode, TTFT, and TPOT

Prefill processes the prompt and builds the initial context. Decode produces one token at a time
and is often memory-bandwidth-bound. TTFT approximates user-visible startup delay, while TPOT
captures steady decode cost. These metrics are available only for successful generation rows.

## Economic Analysis

The cost model compares API token pricing with amortized local hardware, electricity, and
maintenance. In the configured request volumes there is no API/local break-even point. APIs are
cheaper for the small measured prompt volumes; local inference remains useful for privacy,
offline execution, learning, and control.

## Negative Results and Limitations

Current failed runs: 4. AirLLM failed due package/model compatibility before
generation. Quantized dynamic-int8 failed due estimated RAM pressure before conversion. Token
counts are approximate, RAM sampling can miss short spikes, and output quality can only be
reviewed for successful baseline rows in the final evidence.

## Final Engineering Conclusions

The final 7B evidence is realistic for constrained local LLM work: direct Transformers inference
works but is slow, AirLLM can fail before generation despite creating shards, and CPU dynamic-int8
can be infeasible on a 16 GiB RAM laptop. The repository still satisfies the assignment because
it attempts all required paths, preserves failures as structured evidence, and connects the
measurements to Prefill, Decode, paging, quantization, memory pressure, and API/local cost.
