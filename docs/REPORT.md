# Exercise 05 Technical Report

## Executive Summary

This project implements a reproducible local LLM experiment for three execution strategies:
direct Hugging Face Transformers inference, AirLLM-style layer paging, and quantized
inference. The final experiment used `Qwen/Qwen2.5-3B-Instruct` on a Windows laptop with
15.70 GiB RAM and an RTX 3050 Laptop GPU visible to CUDA, two prompts, one run per prompt, and
16 configured max new tokens.

The direct baseline succeeded but was slow. The AirLLM path imported and created layer shards,
then failed inside AirLLM with `IndexError: list index out of range`. The quantized dynamic-int8
path succeeded and was faster at generation than the direct baseline in this short run. The
failed AirLLM run is kept as a valid negative result because it demonstrates the practical
compatibility risk of using a paging framework with current package and model versions.

## Hardware Specification

The hardware snapshot is produced by:

```bash
uv run airllm-ex05 hardware
```

Final observed hardware:

- OS: Windows 11
- CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- Physical cores: 4
- Logical cores: 8
- RAM: 15.70 GiB
- CUDA-visible GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- CUDA-visible VRAM: 4.0 GiB

This is a constrained local environment for a 3B-parameter instruction model. CUDA was visible,
but the laptop GPU has only 4.0 GiB VRAM, so the experiment is still limited by small local
memory budgets, CPU/GPU transfer behavior, and package compatibility. The baseline used the
configured `device: auto` path and recorded CUDA peak allocation, while the final dynamic-int8
quantized path ran as a CPU quantization path.

## Selected Model

Configured model:

```text
Qwen/Qwen2.5-3B-Instruct
```

The model was chosen because it is large enough to stress a 16 GiB laptop with a small 4 GiB
GPU while still being bounded enough for a short local experiment. The final config
intentionally uses only two prompts and `max_new_tokens: 16` so the experiment can complete
without turning the submission into an overnight benchmark.

## Experiment Commands

```bash
uv sync
uv sync --extra models
uv run airllm-ex05 hardware
uv run airllm-ex05 baseline --config configs/experiment.yaml
uv run airllm-ex05 airllm --config configs/experiment.yaml
uv run airllm-ex05 quantized --config configs/experiment.yaml
uv run airllm-ex05 analyze --config configs/experiment.yaml
uv run airllm-ex05 report --config configs/experiment.yaml
```

Each runner writes one JSON result per prompt/run and one CSV summary. Analysis writes a
processed comparison table, analysis JSON, and figures.

## Measurements

Final processed evidence:

- Raw result count: 6
- Successful runs: 4
- Failed runs: 2
- Break-even monthly request volume: none in the configured volume range

Current `results/processed/comparison_table.csv` contains:

| Runner | Status | Prompt | Load s | Latency s | TTFT s | TPOT s/token | Tok/s | Peak RAM MB | Peak VRAM MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| airllm | failed | 0 |  |  |  |  |  |  |  |
| airllm | failed | 1 |  |  |  |  |  |  |  |
| baseline | success | 0 | 45.07 | 24.37 | 7.82 | 1.11 | 0.90 | 4734.62 | 2899.59 |
| baseline | success | 1 | 45.07 | 17.67 | 1.17 | 0.61 | 1.64 | 4744.74 | 2899.70 |
| quantized | success | 0 | 101.56 | 10.18 | 5.48 | 0.41 | 2.46 | 4569.58 | 0.00 |
| quantized | success | 1 | 101.56 | 4.35 | 0.34 | 0.16 | 6.21 | 4592.52 | 0.00 |

The README may contain earlier summarized values if results are regenerated. For final
submission, the authoritative evidence is the committed processed table plus the raw result
files from the local run.

## Baseline Direct Run

The baseline runner uses Transformers `AutoTokenizer` and `AutoModelForCausalLM`. It loads the
model directly and generates with the configured prompts and output limit.

Observed result:

- Both final prompts succeeded.
- Load time was about 45.07 seconds in the current processed table.
- Prompt 0 latency was 24.37 seconds.
- Prompt 1 latency was 17.67 seconds.
- Throughput ranged from 0.90 to 1.64 output tokens per second.
- TTFT varied strongly by prompt, from 1.17 to 7.82 seconds.

Interpretation:

Direct Transformers inference is possible for this bounded 3B run, but it is slow on this
local machine. The decode phase is especially sensitive to memory bandwidth because each output
token requires repeated model weight access. The high TTFT for prompt 0 shows the
user-visible cost of initial prompt processing and first-token production.

## AirLLM Run

AirLLM was included because the lecture frames it as LLM execution through virtual-memory-like
layer paging. The expected advantage is lower resident memory pressure: the system can move
layers through memory instead of keeping the entire model resident. The expected downside is
latency because disk I/O becomes part of the forward path.

Observed result:

- AirLLM dependencies were resolved far enough for the package to import.
- AirLLM created 76 layer-shard files totaling about 6.17 GB.
- AirLLM then failed during its internal loading path.
- Both prompt results were recorded as failed.
- Error type: `IndexError`.
- Error message: `list index out of range`.

Interpretation:

This is a negative compatibility result, not a missing experiment. AirLLM did reach the
disk-sharding stage, which confirms the paging-style setup began. It did not reach generation,
so this run cannot provide AirLLM latency, TTFT, TPOT, throughput, or output-quality numbers.
The result still supports the assignment discussion because it shows a practical risk: a tool
designed to reduce memory pressure can fail before inference because of package, model, or
architecture assumptions.

## Quantization Run

The final successful quantized path used CPU dynamic int8 via PyTorch. Earlier 4-bit
`bitsandbytes` style quantization was not suitable for the observed Windows setup, so the
project keeps that path available but uses dynamic int8 for final validation.

Observed result:

- Both final prompts succeeded.
- Load time was about 101.56 seconds.
- Prompt 0 latency was 10.18 seconds.
- Prompt 1 latency was 4.35 seconds.
- Throughput ranged from 2.46 to 6.21 output tokens per second.
- TTFT ranged from 0.34 to 5.48 seconds.
- Measured process RSS was roughly 4.57-4.59 GB in the current processed table.

Interpretation:

Dynamic-int8 improved generation speed compared with the direct baseline in this local run.
The throughput improvement is consistent with quantization reducing weight precision and memory
bandwidth pressure during decode. However, the measured process RSS did not prove a universal
memory reduction, and load time was higher than baseline in the current evidence. The correct
conclusion is that dynamic-int8 helped runtime throughput on this hardware and configuration,
not that every memory metric improved.

## Prefill, Decode, TTFT, And TPOT

Prefill processes the input prompt and builds the initial context state. It tends to be more
compute-oriented because many prompt tokens can be processed together. Decode produces output
one token at a time. Decode often becomes memory-bandwidth-bound because each new token
requires another pass through model weights.

This project records:

- TTFT: time to first generated token, used as a proxy for user-visible startup delay.
- TPOT: time per output token after generation latency is known.
- Throughput: output tokens per second.
- Total latency: complete generation time for the prompt.

Baseline and quantized runs include streaming TTFT. AirLLM does not include TTFT because it
failed before generation.

## Memory And VRAM

The final machine has a CUDA-visible RTX 3050 Laptop GPU with 4.0 GiB VRAM. The baseline rows
record roughly 2.9 GB CUDA peak allocation, while the dynamic-int8 rows record 0.0 MB VRAM
because that quantized path is CPU-based. CPU process RSS remains the main comparable memory
signal across both successful runners.

Important limitations:

- RAM sampling can miss short peaks.
- Tokenizer/model internals may allocate memory before or after the measured generation window.
- CPU dynamic quantization can improve compute or bandwidth behavior without lowering every
observed process-memory number.
- AirLLM memory benefits cannot be measured for generation because AirLLM failed before
  generating tokens.
- VRAM values should be compared by execution path; baseline and CPU dynamic-int8 do not use
  the same backend.

## Economic Analysis

The cost model compares low-cost API token pricing with local hardware amortization,
electricity, and maintenance. Configured assumptions:

- API input price: 0.15 USD per 1M tokens
- API output price: 0.60 USD per 1M tokens
- Cached input discount: 0.5
- Hardware cost: 1200 USD
- Hardware lifetime: 36 months
- Electricity: 0.20 USD/kWh
- Average power: 180 W
- Maintenance: 10 USD/month
- Monthly volumes: 100, 1000, 10000, 100000 requests

Current analysis found no local/API break-even point in the configured volumes. The local
monthly cost is dominated by hardware amortization and maintenance, while the API cost remains
very small for the short prompts and small request counts used here.

Conclusion:

- API inference is economically better for low-volume or bursty usage in this scenario.
- Local inference may still be justified for privacy, offline use, learning, control, or very
  high sustained volume.
- On this constrained laptop, performance is not competitive with a production GPU service.

## Output Quality

The successful baseline and quantized runs produced text for both prompts. Because the final
experiment uses only two prompts and 16 max new tokens, output quality should be treated as a
small manual spot check rather than a full evaluation. The report should not claim broad model
quality conclusions from this sample.

The relevant assignment conclusion is narrower: dynamic-int8 remained usable for the tested
short explanatory prompts while improving speed on the local CPU.

## Negative Results And Limitations

Negative results:

- AirLLM failed with `IndexError: list index out of range` after shard creation.
- Earlier `bitsandbytes` quantization was unsuitable for the Windows environment.
- The final comparison is not a clean GPU-versus-GPU comparison: baseline recorded CUDA
  allocation, while dynamic-int8 used CPU quantization.

Limitations:

- Token counts are approximate.
- RAM peaks are sampled.
- The experiment is intentionally short.
- Cost power values are assumptions, not watt-meter measurements.
- AirLLM generation performance is unavailable due to failure before generation.
- Package compatibility can change with future versions.

## Final Engineering Conclusions

The project satisfies the assignment as an engineering experiment:

- It documents hardware.
- It attempts direct inference, AirLLM, and quantization.
- It records successes and failures as structured evidence.
- It connects observed behavior to Prefill, Decode, memory pressure, paging, and quantization.
- It generates plots and cost analysis.
- It includes tests and reproducible CLI commands.

The core technical finding is that direct local Transformers inference with Qwen2.5-3B is
possible but slow on this constrained laptop, AirLLM can fail despite successful shard
creation, and CPU dynamic-int8 quantization provides a practical speed improvement for this
bounded local run.
