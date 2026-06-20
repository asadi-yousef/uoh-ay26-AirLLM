# Exercise 05 Technical Report

## Executive Summary

This project now contains a complete local LLM experiment for Exercise 05 using
`Qwen/Qwen2.5-3B-Instruct` on a CPU-only Windows laptop. The model is large enough to make
direct local inference visibly slow on this machine, while still small enough to complete a
bounded experiment with 16 output tokens.

The final result set contains six benchmark records:

- Baseline direct `transformers`: 2 successful runs.
- CPU dynamic-int8 quantization: 2 successful runs.
- AirLLM: 2 failed runs after successful layer-shard creation.

The key engineering result is that direct CPU inference works but is slow: the two baseline
generations took 50.33 seconds and 62.87 seconds, with exact TTFT values of 10.43 seconds and
3.87 seconds. CPU dynamic-int8 quantization reduced generation latency to 11.88 seconds and
6.18 seconds and improved throughput from about 0.45 tokens/second to 3.40 tokens/second on
average. AirLLM created 76 layer-shard files totaling about 6.17 GB, proving that the paging
setup began, but then failed during its internal load path with `IndexError: list index out of
range`.

## Source Requirements

The private PDFs in `documents/` were inspected and remain ignored by Git. The assignment
requires a technical report and README that document hardware, justify the model choice,
attempt direct baseline inference, run AirLLM and quantization on the same task, measure TTFT,
TPOT, throughput, latency, RAM/VRAM, and output quality, and connect the results to Prefill,
Decode, memory pressure, virtual memory, paging, quantization, and API versus on-premises
economics.

## Hardware

The hardware snapshot in `results/raw/hardware.json` reports:

| Field | Value |
| --- | --- |
| CPU | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Physical / logical cores | 4 / 8 |
| RAM | 15.70 GiB |
| GPU / VRAM | None detected |
| CUDA | false |
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.3 |
| Storage C: | NTFS, 932.68 GiB total, 312.45 GiB free |
| Storage G: | FAT32, 932.68 GiB total, 296.82 GiB free |

This is a CPU-only on-premises setup. There is no CUDA acceleration and no VRAM, so the
experiment stresses system RAM, disk cache/offload behavior, CPU throughput, and Python/ML
package compatibility.

## Model Choice

Final model: `Qwen/Qwen2.5-3B-Instruct`.

This model was selected after validating the pipeline with `sshleifer/tiny-gpt2` and trying
`Qwen/Qwen2.5-1.5B-Instruct`. The final choice is stronger for the assignment because:

- It is materially larger than the validation model and stresses CPU-only inference.
- It has a sharded SafeTensors index, which AirLLM requires.
- It is still bounded enough to run on a 15.70 GiB RAM laptop with short output length.
- It is an instruct model, so qualitative output can be inspected more meaningfully than with
  tiny-gpt2.

The experiment used two prompts and `max_new_tokens: 16` to avoid turning the assignment into
an hours-long benchmark.

## Measurement Table

| Runner | Prompt | Status | Load s | Latency s | TTFT s | TPOT s/token | Throughput tok/s | Output tokens | Peak RAM MB | Peak VRAM MB | Failure |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0 | success | 710.513 | 50.329 | 10.435 | 2.288 | 0.437 | 22 | 7737.29 | n/a | n/a |
| baseline | 1 | success | 710.513 | 62.872 | 3.872 | 2.168 | 0.461 | 29 | 7742.39 | n/a | n/a |
| quantized | 0 | success | 159.312 | 11.875 | 5.562 | 0.440 | 2.274 | 27 | 9566.62 | n/a | n/a |
| quantized | 1 | success | 159.312 | 6.183 | 0.442 | 0.221 | 4.529 | 28 | 9573.74 | n/a | n/a |
| airllm | 0 | failed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `IndexError: list index out of range` |
| airllm | 1 | failed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | same as above |

Derived averages from successful runs:

| Metric | Baseline | Quantized dynamic-int8 | Interpretation |
| --- | ---: | ---: | --- |
| Load time | 710.51 s | 159.31 s | Quantized path loaded about 4.46x faster. |
| Generation latency | 56.60 s | 9.03 s | Quantized generation was about 6.27x faster. |
| TTFT | 7.15 s | 3.00 s | Quantized path reduced visible startup delay. |
| TPOT | 2.23 s/token | 0.33 s/token | Quantized decode was about 6.74x faster. |
| Throughput | 0.45 tok/s | 3.40 tok/s | Quantized throughput was about 7.57x higher. |
| Peak process RAM | 7739.84 MB | 9570.18 MB | Dynamic quantization increased total process RSS here. |

## Baseline Analysis

The direct baseline succeeded, but it is slow enough to satisfy the assignment's requirement
for an uncomfortable local run. Initial loading took 710.51 seconds, including model shard
download and checkpoint loading. Transformers/Accelerate reported that some parameters were
offloaded to CPU and disk, which is direct evidence of memory pressure and offload behavior on
this CPU-only laptop.

Generation was also slow:

- Prompt 0: 50.33 seconds total, TTFT 10.43 seconds, TPOT 2.29 seconds/token.
- Prompt 1: 62.87 seconds total, TTFT 3.87 seconds, TPOT 2.17 seconds/token.

The bottleneck is not VRAM, because the machine has no CUDA-visible GPU. It is CPU/RAM/disk
bound: the model can run, but the combination of CPU execution and offload makes both startup
and decode slow.

## AirLLM Analysis

AirLLM was made importable by pinning compatible dependencies:

- `optimum>=1.27,<2`
- `transformers>=4.42,<4.49`
- `sentencepiece>=0.2`

The first AirLLM attempts failed on tiny-gpt2 and Qwen2.5-1.5B because those checkpoints did
not provide the sharded SafeTensors index expected by AirLLM. `Qwen/Qwen2.5-3B-Instruct` does
provide that index. With this model, AirLLM successfully fetched the files and split the model
into 76 layer-shard files under `airllm_cache/layer_shards/splitted_model`, totaling about
6.17 GB.

The actual AirLLM inference still failed during the load path:

```text
IndexError: list index out of range
```

This is a negative result, but it is a useful one. It shows that the paging-style preparation
worked, while the installed AirLLM package did not complete a usable Qwen2.5-3B load on this
Windows/Python environment. The assignment explicitly allows negative results when they are
measured and explained. The engineering conclusion is that AirLLM reduces the theoretical
memory-residency requirement by sharding layers, but package/model/platform compatibility is a
real operational risk.

## Quantization Analysis

The project uses CPU dynamic-int8 quantization through `torch.ao.quantization.quantize_dynamic`
because this machine has no CUDA and Windows `bitsandbytes` support is limited. This is not
the same as 4-bit CUDA quantization, but it is a valid lower-precision local inference path for
the observed hardware.

The quantized path succeeded for both prompts:

- Prompt 0: latency 11.88 seconds, TTFT 5.56 seconds, TPOT 0.44 seconds/token.
- Prompt 1: latency 6.18 seconds, TTFT 0.44 seconds, TPOT 0.22 seconds/token.

Compared with baseline, quantization substantially improved latency, decode speed, and
throughput. The tradeoff is that peak process RSS was higher in this implementation. That can
happen because PyTorch dynamic quantization may keep additional module structures or temporary
copies during conversion; quantization reduced compute/decode cost, but did not reduce total
process memory in this measured run.

Qualitatively, both baseline and quantized outputs are short but relevant. Baseline prompt 0
began explaining Prefill in local LLM inference. Baseline prompt 1 correctly described paging
as using external storage/memory management. Quantized prompt 1 also gave an appropriate
summary. Quantized prompt 0 drifted into a less relevant phrasing, so quality was acceptable
for pipeline evidence but not clearly better than baseline.

## Prefill, Decode, TTFT, and TPOT

Prefill processes the prompt and prepares internal state. In serving metrics, TTFT is the
visible delay until the first generated token and is a practical proxy for prefill plus startup
overheads. Decode emits one token at a time; TPOT captures the steady-state decode cost.

This project now measures TTFT for Transformers-based runners using streaming generation.
The measurements show:

- Baseline TTFT can be several seconds on CPU, especially when the model is offloaded.
- Baseline TPOT around 2.2 seconds/token means decode is very slow.
- Quantized TPOT around 0.33 seconds/token shows a major decode improvement.

The result matches the lecture model: on limited local hardware, decode is strongly affected
by memory movement and CPU throughput. Quantization reduces the amount and cost of work during
generation, so throughput improves.

## Memory, VRAM, and Paging

There is no VRAM measurement because no GPU was detected. All memory pressure is system RAM
and disk/cache behavior.

Baseline peak RSS was about 7.74 GB, and Transformers reported CPU/disk offload. Quantized
peak RSS was about 9.57 GB, which is higher despite faster generation. AirLLM created disk
layer shards totaling about 6.17 GB. These three facts illustrate the central memory tradeoff:

- Direct Transformers can use offload to make a 3B model run, but latency is high.
- Dynamic int8 can speed up CPU generation, but process RSS is not guaranteed to drop.
- AirLLM explicitly materializes layer files and tries to page through them, but the current
  package/model/platform combination failed before generation.

## Figures

The generated figures are:

- `results/figures/latency.png`: baseline versus quantized generation latency.
- `results/figures/throughput.png`: baseline versus quantized throughput.
- `results/figures/memory.png`: baseline versus quantized peak process RAM.
- `results/figures/cost_curve.png`: API versus local monthly cost curve.

The latency and throughput plots show the strongest result: dynamic-int8 quantization made the
3B model much more usable on this CPU-only laptop. The memory plot shows the important caveat:
this quantized path used more process RAM, not less.

## Economic Analysis

Configured assumptions:

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
| 100 | 0.001695 | 43.366148 | 0.00001695 | 0.43366148 |
| 1,000 | 0.016950 | 43.661484 | 0.00001695 | 0.04366148 |
| 10,000 | 0.169500 | 46.614835 | 0.00001695 | 0.00466148 |
| 100,000 | 1.695000 | 76.148354 | 0.00001695 | 0.00076148 |

No break-even point appears in the configured request range. For this small two-prompt
workload, API token pricing is far cheaper than amortized local hardware and maintenance.
Local inference is still justified when privacy, offline execution, regulatory control, or
data residency matters more than direct cost.

Prompt caching strengthens the API side for repeated long prompts because shared context can
be billed at a discount and may avoid repeated Prefill work. That pushes the economic
break-even point further away from local inference for repetitive workloads.

## Limitations

- Only one final model size was benchmarked.
- Output length was intentionally short, `max_new_tokens: 16`, to keep the experiment bounded.
- Power draw was estimated from config, not measured from the wall.
- AirLLM did not complete generation, so its latency/throughput could not be compared.
- The quantized path is CPU dynamic int8, not 4-bit CUDA quantization.
- Peak RAM is process RSS sampled during generation; it may miss short conversion/load peaks.
- Generated raw results and model caches remain ignored by Git.

## Final Conclusions

The final experiment satisfies the core engineering intent of the assignment. A 3B instruct
model runs locally on CPU-only hardware, but direct baseline inference is slow and visibly
resource constrained. Exact TTFT and TPOT measurements show high startup and decode costs.
CPU dynamic-int8 quantization makes the same model much more usable, improving generation
latency by about 6.27x and throughput by about 7.57x, although it increases measured process
RSS in this implementation.

AirLLM partially succeeded: it fetched the sharded model and created 6.17 GB of per-layer
SafeTensors shards, demonstrating the paging-oriented setup. It failed before generation with
an internal index error, which should be reported as a negative compatibility result rather
than hidden.

For this machine and workload, API usage remains cheaper in direct dollars. Local inference is
most defensible for privacy, offline work, and learning how on-premises LLM deployment behaves
under real CPU/RAM/disk constraints.
