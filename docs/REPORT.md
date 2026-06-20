# Exercise 05 Technical Report

## Executive Summary

This repository implements and runs a local LLM benchmarking pipeline for Exercise 05:
direct Hugging Face inference, AirLLM-style inference, and quantized inference. The
assignment asks for a model that is uncomfortable for the local machine and for a technical
analysis of the memory, latency, paging, quantization, and economic tradeoffs. The current
measured run should be interpreted as a pipeline validation run, not as the final stress-model
experiment: the configured model is `sshleifer/tiny-gpt2`, which is intentionally small.

The run produced six raw benchmark records from two prompts and one run per prompt. The
baseline `transformers` runner succeeded twice. The AirLLM runner failed twice during model
loading because `optimum.bettertransformer` was missing. The quantized runner failed twice
during model loading because 4-bit `bitsandbytes` support was unavailable. These failures are
valid negative results for dependency and platform readiness, but they do not yet demonstrate
the assignment's intended large-model RAM/VRAM pressure.

## Source Requirements

The private course PDFs in `documents/` were inspected and remain ignored by Git. The
exercise requires:

- Hardware documentation: CPU, core count, RAM, GPU/VRAM, CUDA availability, OS, Python, and
  storage.
- A Hugging Face model selected to be large or uncomfortable for the local hardware, with
  justification by size, format, and expected memory footprint.
- A direct baseline attempt, then the same task through AirLLM and quantization.
- Metrics: total latency, TTFT, TPOT or ITL, throughput, peak RAM, peak VRAM, load time,
  estimated power/cost, and output quality observations.
- Analysis tied to lecture concepts: Prefill versus Decode, compute-bound versus
  memory-bound behavior, VRAM/RAM pressure, virtual memory, paging, `mmap`, AirLLM layer
  movement, quantization tradeoffs, and on-premises versus API economics.
- Professional software expectations: `uv`, modular package structure, config-driven
  behavior, no secrets or heavy generated files in Git, Ruff, pytest, and documentation.

## Hardware Specification

The hardware snapshot was generated in `results/raw/hardware.json` and is ignored as a
generated experiment artifact.

| Field | Measured value |
| --- | --- |
| CPU | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Physical cores | 4 |
| Logical cores | 8 |
| RAM | 15.70 GiB |
| GPU | None detected |
| VRAM | None detected |
| CUDA | false |
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.3 |
| Storage C: | NTFS, 932.68 GiB total, 312.45 GiB free |
| Storage G: | FAT32, 932.68 GiB total, 296.82 GiB free |

This is CPU-only local inference hardware. The absence of CUDA and VRAM is central: direct
GPU acceleration is unavailable, AirLLM would rely on CPU/disk behavior, and typical
`bitsandbytes` CUDA quantization paths are unlikely to work on this machine without changing
platform, dependencies, or backend.

## Model Choice

Current configured model: `sshleifer/tiny-gpt2`.

This model is suitable for validating the software pipeline because it is public, small, fast
to load, and cheap to test. It is not suitable as the final Exercise 05 stress model because
it does not challenge 15.70 GiB RAM and does not demonstrate the intended large-model failure
or paging behavior. The final manual run should replace it with a larger Hugging Face causal
LM whose unquantized weights are large enough to stress CPU RAM or disk paging, while still
being realistic for a bounded experiment.

The current model also explains the qualitative output: both successful baseline samples
repeat the token "factors" many times. That is expected from a tiny validation model and is
not evidence of useful answer quality.

## Experiment Configuration

The active config in `configs/experiment.yaml` used:

| Setting | Value |
| --- | --- |
| Prompts | 2 prompts about Prefill/Decode and paging |
| Runs | 1 per prompt |
| Max new tokens | 32 |
| Baseline mode | `transformers` |
| AirLLM mode | `airllm` |
| Quantization mode | `bitsandbytes`, 4-bit |
| Model cache | `model_cache/huggingface` |
| AirLLM shard cache | `airllm_cache/layer_shards` |
| Raw outputs | `results/raw` |
| Processed outputs | `results/processed` |
| Figures | `results/figures` |

The experiment commands inferred from the output set are:

```bash
uv run airllm-ex05 hardware
uv run airllm-ex05 baseline --config configs/experiment.yaml
uv run airllm-ex05 airllm --config configs/experiment.yaml
uv run airllm-ex05 quantized --config configs/experiment.yaml
uv run airllm-ex05 analyze --config configs/experiment.yaml
uv run airllm-ex05 report --config configs/experiment.yaml
```

## Raw Result Summary

| Runner | Prompt | Status | Load s | Latency s | TPOT s/token | Throughput tok/s | Output tokens | Peak RAM MB | Peak VRAM MB | Failure |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0 | success | 20.703 | 0.102889 | 0.002393 | 417.924 | 43 | 389.88 | n/a | n/a |
| baseline | 1 | success | 20.703 | 0.051292 | 0.001115 | 896.826 | 46 | 389.95 | n/a | n/a |
| airllm | 0 | failed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `ModuleNotFoundError: No module named 'optimum.bettertransformer'` |
| airllm | 1 | failed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | same as above |
| quantized | 0 | failed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes` |
| quantized | 1 | failed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | same as above |

Baseline average generation latency across the two prompts was about 0.077091 seconds. The
average of the two reported throughput values was about 657.375 tokens/second, while aggregate
throughput across both prompts was about 577.25 tokens/second. Peak process RAM stayed near
389.9 MB. TTFT was not measured because the current runner uses non-streaming generation.

## Baseline Direct Run

The direct Hugging Face baseline succeeded with `sshleifer/tiny-gpt2`. Loading took 20.703
seconds, much longer than either measured generation call. This is expected for a first local
model load on CPU because dependency initialization, cache access, and model construction are
front-loaded.

The successful generation latencies were very small because the model is tiny. This means the
baseline results validate the benchmark plumbing, but they do not establish a direct-inference
bottleneck. There is no evidence in the current result files of out-of-memory behavior,
excessive swap, CUDA pressure, or VRAM pressure during baseline.

Generated samples:

- Prompt 0 output repeated "factors" after the original prompt.
- Prompt 1 output also repeated "factors" after the original prompt.

The qualitative result is poor, but this is a model-choice artifact, not a quantization or
AirLLM finding.

## AirLLM Result

AirLLM failed during load for both prompts:

```text
ModuleNotFoundError: No module named 'optimum.bettertransformer'
```

The failure occurred before inference, so there are no AirLLM latency, throughput, RAM, or
quality measurements. This is a dependency readiness failure rather than a measured AirLLM
performance failure. It still matters for the engineering report: the AirLLM path is optional,
version-sensitive, and depends on packages that may not be available in the default environment.

Conceptually, AirLLM is relevant because it changes the resource allocation model. Instead of
requiring all weights to be resident in RAM or VRAM, it loads model layers progressively from
disk, similar to virtual memory paging. That can make a too-large model runnable on modest
hardware, but it normally increases latency because disk I/O enters the forward pass. The
current run did not reach that stage, so this conclusion remains theoretical for this machine
until the missing dependency is resolved and a stress model is tested.

## Quantized Result

The quantized runner failed during load for both prompts:

```text
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

The configured quantization mode is 4-bit `bitsandbytes`. On this Windows CPU-only setup,
the result is consistent with the known operational risk that common `bitsandbytes`
quantization paths are easiest on CUDA-enabled Linux environments and may be unavailable on
Windows or CPU-only machines. No quantized output was generated, so there is no measured
memory/speed/quality tradeoff yet.

The expected tradeoff remains: quantization reduces weight precision and can reduce memory
footprint and memory bandwidth demand, especially during Decode. The risk is quality loss,
particularly with aggressive low-bit settings or unsupported model/backend combinations.
Because this run failed before loading, the quality red line was not observed.

## Prefill, Decode, TTFT, and TPOT

The lecture separates inference into two phases:

- Prefill: the model processes the prompt and builds internal state such as the KV cache. This
  is more parallel and usually compute-bound on GPU hardware.
- Decode: the model emits one token at a time. This is sequential and often memory-bandwidth
  bound because weights and cached state must be accessed repeatedly.

The current benchmark approximates TPOT as total generation latency divided by output token
count. It does not measure TTFT because `model.generate(...)` is called without token streaming.
Therefore:

- TTFT is missing in the current evidence.
- TPOT is available only for successful baseline runs.
- The measured TPOT values, 2.393 ms/token and 1.115 ms/token, are validation-model numbers
  and should not be generalized to large local LLM serving.

For the final stress experiment, a streaming callback or token-level timing hook should be
added if exact TTFT and inter-token latency are required.

## Memory-Bound Versus Compute-Bound Analysis

The current run does not show a hardware bottleneck. Peak baseline RAM was about 390 MB, far
below the machine's 15.70 GiB RAM. There is no detected GPU or VRAM, so no VRAM saturation can
be analyzed. The successful baseline generation was fast because the model is too small.

For the intended final experiment, the diagnostic logic should be:

- If direct loading fails or the OS starts swapping heavily, the bottleneck is RAM capacity or
  virtual memory pressure.
- If CUDA exists and VRAM fills before generation starts, the bottleneck is VRAM capacity.
- If Prefill dominates TTFT on GPU, the path is likely compute-bound.
- If Decode has high TPOT and low arithmetic utilization, the path is likely memory-bandwidth
  or I/O bound.
- If AirLLM runs but latency increases sharply, the likely cause is layer paging and disk I/O.

## VRAM, RAM, Paging, and AirLLM

This machine has no CUDA-visible GPU. All successful work in the current run used CPU and RAM.
That makes AirLLM's paging idea especially relevant for future runs: if a larger model cannot
fit comfortably in RAM, AirLLM may move the bottleneck from memory capacity to disk bandwidth
and page-cache behavior.

The assignment and lecture compare AirLLM to virtual memory:

- Model layers behave like pages.
- Disk-backed shards behave like a backing store.
- The operating system page cache and `mmap`-style access can reduce explicit copying.
- The price is latency, because page misses and disk reads are much slower than RAM/VRAM.

The current AirLLM failure means no empirical paging measurements were produced. This should be
called out as a limitation, not hidden.

## Figures and Tables

Generated figures exist under `results/figures/`:

- `latency.png`: bar chart for successful baseline latency only.
- `throughput.png`: bar chart for successful baseline throughput only.
- `memory.png`: bar chart for successful baseline peak RAM only.
- `cost_curve.png`: API versus on-premises monthly cost curve from configured assumptions.

Because AirLLM and quantized runs failed before inference, the metric plots contain only
baseline bars. The cost plot is still useful because it is computed from the benchmark-derived
average request shape and configured cost assumptions.

Generated tabular outputs exist under `results/processed/`:

- `analysis.json`: 6 results, 2 successes, 4 failures, no break-even point in configured
  request volumes.
- `comparison_table.csv`: flattened benchmark table with all successes and failures.

These files are ignored by Git and should remain uncommitted. Their important evidence is
summarized in this report.

## Economic Analysis

Configured cost assumptions:

| Assumption | Value |
| --- | ---: |
| API input price | 0.15 USD per 1M tokens |
| API output price | 0.60 USD per 1M tokens |
| Cached input discount | 50% |
| Hardware cost | 1200 USD |
| Hardware lifetime | 36 months |
| Electricity | 0.20 USD/kWh |
| Average local power | 180 W |
| Maintenance | 10 USD/month |

The processed cost curve reports:

| Monthly requests | API cost USD | Local cost USD | API/request USD | Local/request USD |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.002775 | 43.333410 | 0.00002775 | 0.43333410 |
| 1,000 | 0.027750 | 43.334104 | 0.00002775 | 0.04333410 |
| 10,000 | 0.277500 | 43.341042 | 0.00002775 | 0.00433410 |
| 100,000 | 2.775000 | 43.410424 | 0.00002775 | 0.00043410 |

No break-even request volume appears in the configured range. This is unsurprising: the
validation model uses tiny prompts and tiny outputs, so API token costs stay extremely low,
while local cost includes fixed amortized hardware and maintenance. Local execution could
still be justified for privacy, offline operation, regulatory constraints, or very high
volumes, but the current numbers do not justify it economically for this tiny workload.

Prompt/context caching is represented by the cached input discount. For repetitive workloads
with a large shared system prompt or document context, API caching can move the break-even
point further away from local execution by reducing repeated Prefill cost.

## Limitations and Negative Results

Current limitations:

- The model is not the final stress model required by the exercise.
- No GPU or CUDA device was detected, so no VRAM behavior was measured.
- TTFT is not measured because generation is not streamed.
- AirLLM failed before inference because `optimum.bettertransformer` was missing.
- Quantized inference failed before inference because `bitsandbytes` was unavailable.
- Output quality comparison is limited to baseline tiny-gpt2 samples; AirLLM and quantized
  outputs do not exist.
- The cost model is deterministic and useful for comparison, but it uses configured
  assumptions rather than measured wall-power data.

Negative results:

- AirLLM path: failed at load stage for both prompts.
- Quantized path: failed at load stage for both prompts.
- Baseline did not fail, but it also did not demonstrate the intended large-model bottleneck.

## Engineering Conclusions

The software pipeline is in good shape for the assignment workflow: it collects hardware,
loads YAML config, runs the three execution modes, serializes structured successes and
failures, writes CSV/JSON analysis, generates plots, and produces a Markdown report. The
ignored generated outputs contain enough evidence to document the validation run without
committing raw experiment files.

The experiment itself is not yet final-submission complete. It proves the code path, but not
the assignment's central hardware stress claim. Before final manual submission, the project
needs one more real experiment pass with a model selected to stress this CPU-only, 15.70 GiB
RAM machine, plus either a working AirLLM dependency set or a clearly documented reason why
AirLLM cannot run on the target environment. Quantized inference also needs either a supported
backend or a documented platform limitation.

The most defensible final conclusion from current evidence is narrow: the repository is a
reproducible measurement framework, the baseline validation run works, and the AirLLM and
quantized paths expose concrete dependency/platform blockers. The larger performance,
paging, and quantization claims still require a final stress-model run.
