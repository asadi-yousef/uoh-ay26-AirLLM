# Exercise 05 Technical Report

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

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Raw result count: 6
- Successful runs: 6
- Failed runs: 0
- Break-even monthly request volume: None
- Baseline rows: 2
- AirLLM rows: 2
- Quantized rows: 2

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

## Negative Results and Limitations

Current failed runs: 0. Token counts are approximate, RAM sampling can
miss short spikes, AirLLM does not expose TTFT through the same streaming interface, and
bitsandbytes offload depends on CUDA, Accelerate, and Transformers compatibility.

## Final Engineering Conclusions

The final 7B evidence is realistic for constrained local LLM work: direct Transformers inference
works but is slow, AirLLM succeeds but is the slowest generation path on this machine, and
bitsandbytes 8-bit quantization with CPU offload gives the fastest observed generation among the
three tested runners. The repository connects the measurements to Prefill, Decode, paging,
quantization, memory pressure, and API/local cost.
