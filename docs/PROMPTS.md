# Prompt Book

This file records the prompts that matter for the submission. It separates benchmark prompts,
which are real experimental inputs, from representative AI-assistance prompts, which describe
the development workflow without exposing private chat logs, secrets, course PDFs, or tokens.

## Final Benchmark Prompts

The benchmark prompts are stored in `configs/experiment.yaml` and are copied here for review.

| Prompt index | Prompt | Purpose |
| ---: | --- | --- |
| 0 | `Explain the difference between prefill and decode in local LLM inference.` | Tests Prefill, Decode, TTFT, TPOT, local inference latency, and first-token behavior. |
| 1 | `Summarize why paging can help run a model that does not fit in VRAM.` | Tests paging, virtual memory, AirLLM motivation, RAM/VRAM pressure, and local model limits. |

Benchmark settings that control prompt execution:

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Runs per prompt: `1`
- Warmup runs: `0`
- Max new tokens: `16`
- Timeout: `900` seconds
- Raw output path: `results/raw/`
- Processed output path: `results/processed/`

## Final Observed Prompt Outcomes

| Runner | Prompt 0 | Prompt 1 |
| --- | --- | --- |
| `baseline` | Success; slow generation with measured TTFT/TPOT/latency/throughput. | Success; slow generation with measured TTFT/TPOT/latency/throughput. |
| `airllm` | Failed before generation with `AttributeError`. | Failed before generation with `AttributeError`. |
| `quantized` | Failed at load stage with dynamic-int8 RAM guard `MemoryError`. | Failed at load stage with dynamic-int8 RAM guard `MemoryError`. |

## Representative Development Prompts

These are reconstructed prompts that match the work performed in this repository:

- Create a professional Exercise 05 repository for local LLM benchmarking with `uv`,
  `pyproject.toml`, modular Python code, tests, docs, plots, and a cost model.
- Translate the assignment into PRDs, an implementation plan, a TODO ledger, and an auditable
  README.
- Design a config-driven benchmark harness that can run baseline Transformers inference,
  AirLLM-style paging, and quantized inference from the same YAML file.
- Implement Pydantic config models and result schemas for hardware snapshots, metrics, generated
  text, and structured failures.
- Add a CLI with `hardware`, `baseline`, `airllm`, `quantized`, `analyze`, and `report`
  commands.
- Implement baseline generation with `AutoTokenizer`, `AutoModelForCausalLM`, streaming TTFT,
  TPOT, throughput, process RAM sampling, and CUDA peak VRAM measurement.
- Implement an AirLLM runner that passes a configured layer-shard path and preserves load or
  generation errors as failed result rows.
- Implement quantization with a `bitsandbytes` path for supported environments and CPU
  `torch.dynamic_int8` for Windows validation.
- Make model packages lazy imports so tests can run without downloading large checkpoints.
- Add fake model/tokenizer tests so CI and local verification stay lightweight.
- Generate comparison CSV files, analysis JSON, latency/throughput/memory plots, and a cost
  curve.
- Explain Prefill, Decode, TTFT, TPOT, throughput, RAM, VRAM, paging, quantization, and API/local
  cost tradeoffs in the README and report.
- Review every generated result and make sure failed runs are described as evidence, not hidden.
- Update the final model from an earlier 3B experiment to `Qwen/Qwen2.5-7B-Instruct`.
- Remove stale 3B Hugging Face cache and old AirLLM shards without deleting the current 7B cache.
- Fix `uv run airllm-ex05 quantized --config configs/experiment.yaml` so an interrupted or
  failed run does not leave stale 3B quantized results behind.
- Add CLI cleanup so each runner removes its old raw JSON/CSV outputs before writing fresh
  results.
- Diagnose why 7B dynamic-int8 stalls or crashes on a 16 GiB RAM laptop.
- Add a memory guard that estimates cached checkpoint size and stops dynamic-int8 before native
  conversion can exceed RAM.
- Regenerate analysis so `results/processed/comparison_table.csv` contains only
  `Qwen/Qwen2.5-7B-Instruct` rows.
- Update README, PRDs, TODO, and report so they no longer claim old 3B or stale quantized-success
  evidence.
- Ensure every Python file has at most 150 lines by splitting focused tests into separate files.
- Run `uv run pytest` and `uv run ruff check .` before final submission.

## Prompting Policy

- Final measurement prompts must live in `configs/experiment.yaml`.
- Representative development prompts may be summarized here when exact chat history is not
  available.
- No secrets, API keys, Hugging Face tokens, private PDFs, model weights, or raw private course
  material should be committed.
- Any future prompt changes must be reflected in both this file and the config before rerunning
  the benchmark.
