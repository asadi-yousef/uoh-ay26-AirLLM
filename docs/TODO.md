# TODO

## Completed

- [x] Inspect assignment, lecture, and software guideline PDFs.
- [x] Create project skeleton with `uv`, `pyproject.toml`, source package, tests, docs, and
      ignored result directories.
- [x] Implement YAML config loading and root-relative path resolution.
- [x] Implement hardware collection.
- [x] Implement benchmark result schemas and JSON/CSV serialization.
- [x] Implement baseline, AirLLM, and quantized runners with structured failure handling.
- [x] Add CPU dynamic-int8 quantization support for Windows CPU-only validation.
- [x] Implement CLI commands for hardware, runners, analysis, and report generation.
- [x] Implement cost model and plot generation.
- [x] Add tests that avoid model downloads and cover the core infrastructure.
- [x] Run the validation experiment with `sshleifer/tiny-gpt2`.
- [x] Summarize ignored raw result evidence in `docs/REPORT.md`.

## Current Experiment Evidence

- [x] Hardware collected: CPU-only Windows machine, 4 physical cores, 8 logical cores,
      15.70 GiB RAM, no CUDA-visible GPU/VRAM.
- [x] Baseline succeeded for both validation prompts.
- [x] AirLLM failed during load because `optimum.bettertransformer` was missing.
- [x] Previous quantized 4-bit path failed during load because `bitsandbytes` was unavailable.
- [x] Current quantized CPU dynamic-int8 path succeeded for both validation prompts.
- [x] Analysis JSON, comparison CSV, and four plots were generated locally and remain ignored.

## Still Needed For Final Course Submission

- [ ] Manually choose a final stress model large enough to challenge 15.70 GiB RAM while still
      being realistic for a bounded local experiment.
- [ ] Rerun baseline, AirLLM, and quantized modes with the final stress model.
- [ ] Resolve or document the AirLLM dependency issue for the final environment.
- [x] Resolve the quantization backend issue for validation by using CPU dynamic int8.
- [ ] Add exact TTFT measurement if the final report needs token-stream timing instead of
      derived TPOT only.
- [ ] Review generated samples manually for output quality and document any quantization
      quality red line.
- [ ] Decide whether the README itself must embed the final plots for the evaluator, or whether
      linking to `docs/REPORT.md` is acceptable.
- [ ] Do a final human read-through of `docs/REPORT.md` after the stress-model run.

## Manual Review Checklist

- [ ] Confirm private PDFs remain ignored and unstaged.
- [ ] Confirm model caches, AirLLM shards, virtual environments, and generated heavy outputs
      remain unstaged.
- [ ] Confirm final report numbers match ignored raw JSON/CSV files.
- [ ] Confirm all stated failures are real and not inferred.
- [ ] Confirm Ruff and pytest pass on the final committed state.
