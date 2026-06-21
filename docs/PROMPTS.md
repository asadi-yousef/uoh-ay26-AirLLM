# Prompt Book

## Experiment Prompts

The final benchmark uses two prompts chosen to target the lecture concepts directly.

| Prompt index | Prompt | Purpose |
| ---: | --- | --- |
| 0 | `Explain the difference between prefill and decode in local LLM inference.` | Measures an explanation task tied to Prefill, Decode, TTFT, and TPOT. |
| 1 | `Summarize why paging can help run a model that does not fit in VRAM.` | Measures an explanation task tied to virtual memory, paging, AirLLM, RAM, and VRAM. |

## AI-Assisted Development Prompts

The project was developed with AI assistance, but the final repository is designed to be
auditable through code, tests, config, and committed result artifacts.

Representative development prompts:

- Build a professional Exercise 05 repository with `uv`, `pyproject.toml`, modular source,
  config-driven runners, tests, docs, plots, and cost analysis.
- Implement baseline, AirLLM, and quantized inference runners that preserve failures as
  structured benchmark rows.
- Generate a report that connects measurements to Prefill, Decode, TTFT, TPOT, VRAM, paging,
  quantization, and API versus local economics.
- Review the docs against the assignment and software-submission PDFs, then align README and
  report with the current processed results.

## Prompting Policy

- Prompts that drive final measurements are stored in `configs/experiment.yaml`.
- Development prompts are summarized here rather than stored with private chat logs.
- No secrets, API keys, Hugging Face tokens, or private course PDFs are committed.
