# TODO

## Completed

- [x] Inspect assignment, lecture, and software guideline PDFs.
- [x] Create project skeleton with `uv`, `pyproject.toml`, source package, tests, docs, and
      ignored result directories.
- [x] Implement YAML config loading and root-relative path resolution.
- [x] Implement hardware collection.
- [x] Implement benchmark result schemas and JSON/CSV serialization.
- [x] Implement baseline, AirLLM, and quantized runners with structured failure handling.
- [x] Add CPU dynamic-int8 quantization support for Windows CPU-only hardware.
- [x] Add streaming TTFT measurement for Transformers baseline and quantized modes.
- [x] Implement CLI commands for hardware, runners, analysis, and report generation.
- [x] Implement cost model and plot generation.
- [x] Add tests that avoid model downloads and cover the core infrastructure.
- [x] Run the validation experiment with `sshleifer/tiny-gpt2`.
- [x] Run the final bounded stress experiment with `Qwen/Qwen2.5-3B-Instruct`.
- [x] Summarize raw result evidence in `docs/REPORT.md` and `README.md`.

## Current Experiment Evidence

- [x] Hardware collected: CPU-only Windows machine, 4 physical cores, 8 logical cores,
      15.70 GiB RAM, no CUDA-visible GPU/VRAM.
- [x] Baseline succeeded for both final prompts but was slow: average latency 56.60 seconds.
- [x] AirLLM dependencies were fixed enough to import and shard the model.
- [x] AirLLM created 76 layer-shard files totaling about 6.17 GB, then failed with
      `IndexError: list index out of range`.
- [x] Previous quantized 4-bit path failed during load because `bitsandbytes` was unavailable.
- [x] Current quantized CPU dynamic-int8 path succeeded for both final prompts.
- [x] Analysis JSON, comparison CSV, and four plots were generated locally.

## Final Course Submission Status

- [x] Manually choose a final stress model large enough to challenge 15.70 GiB RAM while still
      being realistic for a bounded local experiment.
- [x] Rerun baseline, AirLLM, and quantized modes with the final stress model.
- [x] Resolve dependency-level AirLLM issues and document the remaining AirLLM runtime failure.
- [x] Resolve the quantization backend issue for validation by using CPU dynamic int8.
- [x] Add exact TTFT measurement if the final report needs token-stream timing instead of
      derived TPOT only.
- [x] Review generated samples manually for output quality and document any quantization
      quality red line.
- [x] Decide whether the README itself must embed the final plots for the evaluator, or whether
      linking to `docs/REPORT.md` is acceptable.
- [ ] Do a final human read-through of `docs/REPORT.md` after the stress-model run.

## Manual Review Checklist

- [ ] Confirm private PDFs remain ignored and unstaged.
- [ ] Confirm model caches, AirLLM shards, virtual environments, and generated heavy outputs
      remain unstaged.
- [ ] Confirm final report numbers match ignored raw JSON/CSV files.
- [ ] Confirm all stated failures are real and not inferred.
- [ ] Confirm Ruff and pytest pass on the final committed state.
