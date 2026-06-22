# TODO

This file is the detailed work ledger for the Exercise 05 AirLLM project. It intentionally
contains both completed work and remaining manual checks, because the project was planned as a
full engineering submission rather than a single script. Negative experiment outcomes are kept
as planned evidence, not removed from the record.

## 0. Source Material And Assignment Interpretation

- [x] Read the Exercise 05 assignment PDF.
- [x] Read the course lecture notes about local LLM execution.
- [x] Read the software-guidelines PDF.
- [x] Identify that the assignment expects a model that is uncomfortable for local hardware.
- [x] Identify that direct local inference, AirLLM, and quantization all need to be attempted.
- [x] Identify that failure is acceptable when the failure is measured, preserved, and analyzed.
- [x] Extract the required performance vocabulary: Prefill, Decode, TTFT, TPOT, ITL, throughput,
      latency, RAM, VRAM, CPU, GPU, paging, `mmap`, quantization, cost, and quality.
- [x] Decide that the project should be reproducible from a CLI and config file.
- [x] Decide that the final submission should include source code, docs, tests, plots, and tables.
- [x] Decide that raw model outputs and model weights must not be committed.
- [x] Decide that private PDFs must remain untracked.
- [x] Translate the assignment into PRDs, an implementation plan, and this task ledger.

## 1. Repository And Tooling Setup

- [x] Create the Python package layout under `src/airllm_ex05/`.
- [x] Create `pyproject.toml`.
- [x] Set project name to `airllm-ex05`.
- [x] Set package version to `0.1.0`.
- [x] Add a concise project description.
- [x] Set `README.md` as the project readme.
- [x] Allow Python `>=3.11,<3.14`.
- [x] Add core dependencies: `matplotlib`, `pandas`, `psutil`, `pydantic`, `pyyaml`, and `torch`.
- [x] Add optional model dependencies under the `models` extra.
- [x] Add `accelerate` to optional model dependencies.
- [x] Add `airllm` to optional model dependencies.
- [x] Add `bitsandbytes` only for non-Windows platforms.
- [x] Add `optimum`, `sentencepiece`, and `transformers`.
- [x] Pin `transformers` below `4.49` to reduce package-compatibility risk.
- [x] Add dev dependencies: `pytest`, `pytest-cov`, and `ruff`.
- [x] Configure the PyTorch CUDA wheel index for `uv`.
- [x] Add the console script `airllm-ex05`.
- [x] Configure the Hatchling build backend.
- [x] Configure Ruff line length at 100.
- [x] Configure Ruff lint rule families.
- [x] Ignore annotation rules in tests where full typing adds noise.
- [x] Allow the generated report code to contain long Markdown template lines.
- [x] Configure pytest to use `tests/`.
- [x] Configure pytest to include `src/` on `pythonpath`.
- [x] Configure pytest coverage for `airllm_ex05`.
- [x] Set the minimum coverage threshold to 85 percent.
- [x] Put pytest cache under `.tmp/pytest_cache`.

## 2. Git Hygiene And Artifact Policy

- [x] Keep virtual environments out of Git.
- [x] Keep Python bytecode and pytest cache out of Git.
- [x] Keep Hugging Face model cache out of Git.
- [x] Keep AirLLM layer shards out of Git.
- [x] Keep raw per-run JSON files out of the intended final commit unless the evaluator requires
      them explicitly.
- [x] Keep private course PDFs out of Git.
- [x] Keep secrets and Hugging Face tokens out of code and config.
- [x] Commit small generated figures because the README needs visible evidence.
- [x] Commit `results/processed/comparison_table.csv` because it is small final evidence.
- [x] Keep `.gitkeep` files so empty result directories are visible.
- [x] Document artifact policy in the README.
- [x] Document artifact policy in the PRD.
- [x] Document artifact policy in the plan.
- [ ] Before final submission, confirm ignored raw artifacts are not accidentally staged.
- [ ] Before final submission, confirm private PDFs are not staged.
- [ ] Before final submission, confirm model cache directories are not staged.
- [ ] Before final submission, confirm AirLLM shard directories are not staged.

## 3. Documentation Plan

- [x] Create `docs/PRD.md`.
- [x] Create `docs/PLAN.md`.
- [x] Create `docs/PRD_airllm_pipeline.md`.
- [x] Create `docs/PRD_benchmarking.md`.
- [x] Create `docs/PRD_quantization.md`.
- [x] Create `docs/REPORT.md`.
- [x] Create `docs/ARCHITECTURE.md`.
- [x] Create `docs/PROMPTS.md`.
- [x] Create `.env.example`.
- [x] Create `LICENSE` for implementation code.
- [x] Create `docs/screenshots/` for evaluator-facing screenshots.
- [x] Create `docs/TODO.md`.
- [x] Keep the README as the first evaluator-facing entry point.
- [x] Describe project structure in the README.
- [x] Describe installation with `uv`.
- [x] Describe optional model dependency installation.
- [x] Document all CLI commands.
- [x] Document final config settings.
- [x] Document output file locations.
- [x] Document final experiment results.
- [x] Embed final comparison table in the README.
- [x] Embed final plots in the README.
- [x] Document troubleshooting for missing `transformers`.
- [x] Document troubleshooting for AirLLM incompatibility.
- [x] Document troubleshooting for missing CUDA/GPU.
- [x] Document troubleshooting for out-of-memory risk.
- [x] Document troubleshooting for disk pressure.
- [x] Document troubleshooting for Windows quantization limitations.
- [x] Expand docs after implementation so they represent what was actually planned and done.
- [ ] Do a final human read-through of every Markdown document.
- [ ] Confirm links and image paths render correctly on GitHub.
- [ ] Capture final screenshots under `docs/screenshots/` and embed them in README before
      submission.

## 4. Configuration System

- [x] Create `configs/experiment.yaml`.
- [x] Define `model.name`.
- [x] Define `model.cache_dir`.
- [x] Define `model.trust_remote_code`.
- [x] Define `model.device`.
- [x] Define benchmark prompts.
- [x] Define `benchmark.max_new_tokens`.
- [x] Define `benchmark.runs`.
- [x] Define `benchmark.warmup_runs`.
- [x] Define `benchmark.timeout_seconds`.
- [x] Define raw, processed, figure, and report output paths.
- [x] Define baseline runner settings.
- [x] Define AirLLM runner settings.
- [x] Define quantization runner settings.
- [x] Define cost-analysis assumptions.
- [x] Implement typed Pydantic config models.
- [x] Implement root-relative path resolution.
- [x] Resolve model cache paths relative to the repo root.
- [x] Resolve output directories relative to the repo root.
- [x] Resolve AirLLM layer-shard paths relative to the repo root.
- [x] Resolve report output path relative to the repo root.
- [x] Validate config through tests.
- [x] Keep config file human-editable.
- [x] Keep final config bounded enough for a short constrained-local run.
- [x] Set final stress model to `Qwen/Qwen2.5-7B-Instruct`.
- [x] Set final prompts to one Prefill/Decode prompt and one paging prompt.
- [x] Set final max new tokens to 16.
- [x] Set final run count to 1.
- [x] Set final AirLLM shard path to `airllm_cache/qwen2_5_7b/layer_shards`.
- [x] Set final quantization mode to `bitsandbytes`.
- [x] Use 8-bit bitsandbytes with fp32 CPU offload for the final 7B run.
- [x] Preserve `dynamic_int8` for smaller CPU validation models.

## 5. Hardware Collection

- [x] Implement hardware snapshot collection.
- [x] Record OS name and version.
- [x] Record Python version.
- [x] Record CPU name where available.
- [x] Record physical core count.
- [x] Record logical core count.
- [x] Record total RAM in GiB.
- [x] Detect CUDA availability through torch when available.
- [x] Record GPU name when CUDA is visible.
- [x] Record VRAM when CUDA is visible.
- [x] Record no-GPU state for CPU-only hardware.
- [x] Record storage information.
- [x] Write hardware snapshot to `results/raw/hardware.json`.
- [x] Add CLI command for hardware collection.
- [x] Add tests for hardware schema behavior.
- [x] Run final hardware snapshot on Windows.
- [x] Record final hardware as Windows 11.
- [x] Record final constrained Windows laptop environment.
- [x] Record final 4 physical cores.
- [x] Record final 8 logical cores.
- [x] Record final 15.70 GiB RAM.
- [x] Record final NVIDIA GeForce RTX 3050 Laptop GPU.
- [x] Record final 4.0 GiB CUDA-visible VRAM.

## 6. Data Models And Result Contracts

- [x] Define a benchmark metrics schema.
- [x] Define a benchmark result schema.
- [x] Define a hardware schema.
- [x] Include runner name in every result.
- [x] Include model name in every result.
- [x] Include prompt index in every result.
- [x] Include run index in every result.
- [x] Include status in every result.
- [x] Include input token estimate where available.
- [x] Include output token estimate where available.
- [x] Include load time where available.
- [x] Include total latency where available.
- [x] Include TTFT where available.
- [x] Include TPOT where available.
- [x] Include tokens per second where available.
- [x] Include peak RAM where available.
- [x] Include peak VRAM where available.
- [x] Include generated sample text for successful runs.
- [x] Include error type for failures.
- [x] Include error message for failures.
- [x] Include metadata for runner-specific details.
- [x] Serialize result models to JSON.
- [x] Flatten result models to CSV.
- [x] Load benchmark results while ignoring `hardware.json`.
- [x] Preserve mixed success and failure results in analysis.

## 7. Benchmarking Infrastructure

- [x] Implement prompt/run iteration.
- [x] Make every runner use the same prompt/run grid.
- [x] Implement JSON result saving.
- [x] Implement CSV summary saving.
- [x] Implement result loading for analysis.
- [x] Create one JSON file per prompt/run.
- [x] Create one CSV file per runner.
- [x] Capture failures without crashing the entire benchmark.
- [x] Represent load-stage failures for every prompt/run.
- [x] Represent generation-stage failures for the specific prompt/run.
- [x] Add tests for result round trips.
- [x] Add tests for CSV output.
- [x] Add tests for empty result directories.
- [x] Add tests for mixed result loading.
- [x] Keep generated output text capped enough to inspect.

## 8. Metrics

- [x] Implement approximate token counting.
- [x] Derive output token count for generated text.
- [x] Derive TPOT from latency and output tokens.
- [x] Derive throughput as output tokens per second.
- [x] Implement process RSS sampling.
- [x] Implement CUDA peak memory helper.
- [x] Reset CUDA peak memory when available.
- [x] Return zero or null VRAM values when CUDA is unavailable.
- [x] Add streaming TTFT measurement for baseline generation.
- [x] Add streaming TTFT measurement for dynamic-int8 quantized generation.
- [x] Preserve TTFT as unavailable when AirLLM fails before generation.
- [x] Test token counting.
- [x] Test throughput calculation.
- [x] Test TPOT calculation.
- [x] Test RAM sampler behavior.
- [x] Document that simple token counting is approximate.

## 9. Baseline Runner

- [x] Implement direct `transformers` baseline runner.
- [x] Lazy-import `transformers` and torch.
- [x] Load tokenizer with `AutoTokenizer`.
- [x] Load model with `AutoModelForCausalLM`.
- [x] Respect configured cache directory.
- [x] Respect `trust_remote_code`.
- [x] Choose device behavior from config.
- [x] Use a streamer where available to capture TTFT.
- [x] Generate with configured `max_new_tokens`.
- [x] Measure load time.
- [x] Measure total generation latency.
- [x] Measure peak RAM.
- [x] Measure CUDA peak VRAM when available.
- [x] Save generated samples.
- [x] Save structured failures when dependencies or model loading fail.
- [x] Add tests with fake model/tokenizer objects.
- [x] Run final baseline on Qwen2.5-7B.
- [x] Record two successful baseline results.
- [x] Record baseline prompt 0 success.
- [x] Record baseline prompt 1 success.
- [x] Record baseline as slow on constrained local hardware.

## 10. AirLLM Runner

- [x] Implement AirLLM runner.
- [x] Lazy-import AirLLM.
- [x] Prefer `airllm.AutoModel` when available.
- [x] Preserve fallback path for `AirLLMLlama2`.
- [x] Pass configured layer-shard path.
- [x] Use the same prompt/run grid as baseline.
- [x] Save one failed result per prompt when AirLLM load fails.
- [x] Include load-stage metadata on failures.
- [x] Keep AirLLM shards out of Git.
- [x] Add tests for missing AirLLM dependency.
- [x] Add tests for fake AirLLM success.
- [x] Run final AirLLM command on Qwen2.5-7B.
- [x] Resolve dependency issues far enough for AirLLM to import.
- [x] Confirm AirLLM created layer shards.
- [x] Diagnose the AirLLM raw-string generation failure.
- [x] Fix AirLLM by tokenizing prompts before generation.
- [x] Pin Transformers to a compatible version for the AirLLM/Qwen2 stack.
- [x] Record final AirLLM successful generation rows.
- [x] Explain AirLLM through paging and disk I/O in report docs.

## 11. Quantized Runner

- [x] Implement quantized runner.
- [x] Support `bitsandbytes` configuration path.
- [x] Support 4-bit settings.
- [x] Support 8-bit settings.
- [x] Support CPU dynamic-int8 path.
- [x] Use `BitsAndBytesConfig` when configured.
- [x] Use `torch.ao.quantization.quantize_dynamic` for dynamic int8.
- [x] Lazy-import heavy dependencies.
- [x] Reuse the baseline generation helper.
- [x] Include quantization metadata in results.
- [x] Capture missing `bitsandbytes` as structured failure.
- [x] Document the previous dynamic-int8 7B RAM-pressure failure.
- [x] Switch final config to bitsandbytes 8-bit with CPU offload.
- [x] Run final bitsandbytes quantized benchmark.
- [x] Record two successful quantized results for the final 7B run.
- [x] Document quantized latency for the final 7B run.
- [x] Document quantized throughput for the final 7B run.
- [x] Document quantized RAM/VRAM generation metrics for the final 7B run.
- [x] Document quantized output quality smoke-check evidence.
- [x] Keep the dynamic-int8 memory guard for smaller CPU validation and oversized conversions.

## 12. CLI

- [x] Implement `airllm-ex05 hardware`.
- [x] Implement `airllm-ex05 baseline --config configs/experiment.yaml`.
- [x] Implement `airllm-ex05 airllm --config configs/experiment.yaml`.
- [x] Implement `airllm-ex05 quantized --config configs/experiment.yaml`.
- [x] Implement `airllm-ex05 analyze --config configs/experiment.yaml`.
- [x] Implement `airllm-ex05 report --config configs/experiment.yaml`.
- [x] Add CLI smoke tests.
- [x] Keep CLI commands simple enough for evaluator reproduction.
- [x] Document CLI commands in README and docs.
- [x] Keep runner commands independent so a failed AirLLM run does not block analysis.

## 13. Analysis And Plotting

- [x] Implement processed comparison table generation.
- [x] Implement analysis JSON generation.
- [x] Count raw benchmark results.
- [x] Count successful runs.
- [x] Count failed runs.
- [x] Preserve failed rows in the comparison table.
- [x] Generate latency plot.
- [x] Generate throughput plot.
- [x] Generate memory plot.
- [x] Generate cost curve plot.
- [x] Handle missing metric values gracefully.
- [x] Commit final figures for README visibility.
- [x] Commit final comparison table for tabular evidence.
- [x] Add tests for report and analysis generation.
- [x] Verify final analysis count: 6 raw results.
- [x] Verify final success count: 6.
- [x] Verify final failure count: 0.
- [x] Verify final cost analysis has no break-even in configured volumes.

## 14. Cost Model

- [x] Implement API input token price assumption.
- [x] Implement API output token price assumption.
- [x] Implement cached-input discount assumption.
- [x] Implement hardware amortization.
- [x] Implement hardware lifetime in months.
- [x] Implement electricity price assumption.
- [x] Implement average power assumption.
- [x] Implement maintenance assumption.
- [x] Implement monthly request volume scan.
- [x] Compute API cost per volume.
- [x] Compute local cost per volume.
- [x] Compute cost per request.
- [x] Compute break-even volume when one exists.
- [x] Return null break-even when local never beats API in configured range.
- [x] Add cost-model tests.
- [x] Document that low-volume API use is cheaper in the final configured scenario.

## 15. Final Experiment Evidence

- [x] Use `Qwen/Qwen2.5-7B-Instruct`.
- [x] Use two prompts.
- [x] Use one run per prompt.
- [x] Use 16 max new tokens.
- [x] Run hardware collection.
- [x] Run baseline.
- [x] Run AirLLM.
- [x] Run quantized.
- [x] Run analysis.
- [x] Generate report.
- [x] Generate README plots.
- [x] Record final comparison CSV.
- [x] Record AirLLM successes as raw JSON.
- [x] Record baseline successes as raw JSON.
- [x] Record quantized successes as raw JSON.
- [x] Record final baseline load time from processed evidence.
- [x] Record final quantized performance from processed evidence.
- [x] Record final AirLLM performance from raw evidence.
- [x] Record final cost-analysis output.
- [x] Preserve the fact that raw evidence changed between runs if regenerated.
- [ ] Before submission, rerun all commands only if the evaluator requires fresh raw outputs.

## 16. Testing

- [x] Add `tests/test_config.py`.
- [x] Add `tests/test_hardware.py`.
- [x] Add `tests/test_metrics.py`.
- [x] Add `tests/test_benchmark.py`.
- [x] Add `tests/test_runners.py`.
- [x] Add `tests/test_cli.py`.
- [x] Add `tests/test_cost_analysis.py`.
- [x] Add `tests/test_report.py`.
- [x] Add `tests/test_shared_utils.py`.
- [x] Add `tests/test_misc_coverage.py`.
- [x] Mock heavy model dependencies in tests.
- [x] Avoid model downloads in tests.
- [x] Test config path resolution.
- [x] Test hardware model serialization.
- [x] Test metric calculations.
- [x] Test benchmark serialization.
- [x] Test failure result construction.
- [x] Test fake runner success.
- [x] Test fake missing dependency behavior.
- [x] Test CLI smoke paths.
- [x] Test analysis/report output.
- [x] Run `uv run ruff check .`.
- [x] Run `uv run pytest`.
- [x] Re-run Ruff after this documentation expansion.
- [x] Re-run pytest after this documentation expansion.
- [x] Record that pytest needs a workspace `--basetemp` on this machine because the default
      AppData temp path is permission-denied.
- [x] Record that the current full pytest run reports 35 passed with 86.12 percent coverage.

## 17. Report Content Still To Verify Manually

- [x] Confirm all final numeric values in README match `results/processed/comparison_table.csv`.
- [x] Confirm all final numeric values in `docs/REPORT.md` match current raw/processed files.
- [x] Confirm AirLLM shard count and size are documented as observed local evidence.
- [x] Confirm AirLLM generated text claims match the final successful run.
- [x] Confirm quantized speed claims are tied to bitsandbytes 8-bit evidence, not dynamic-int8.
- [x] Confirm output quality observations are based on actual generated text.
- [x] Confirm cost conclusions are tied to configured assumptions.
- [x] Confirm historical negative attempts are described as fixed or superseded, not final
      evidence.
- [x] Confirm all planned assignment terms are discussed in the final report.
- [x] Confirm the README is enough for a fast evaluator scan.

## 18. Known Limitations

- [x] Token counting is approximate rather than tokenizer-exact.
- [x] AirLLM TTFT is unavailable because AirLLM does not expose the same streaming callback path.
- [x] Peak RAM sampling can miss short-lived memory spikes.
- [x] CUDA VRAM metrics are available for paths that use CUDA/offload.
- [x] `bitsandbytes` is the final quantization backend for the 7B run.
- [x] CPU dynamic-int8 is implemented but reserved for smaller CPU validation models.
- [x] Python 3.12 was used for the final model run; Python 3.13 may not be ideal for every ML
      package even though tooling can run.
- [x] The final experiment is intentionally bounded and short to complete on local hardware.
- [x] Cost analysis uses assumptions rather than measured wall-plug power.
- [x] Output quality review is manual and small-sample.
- [x] AirLLM compatibility is version-sensitive; the current lock uses Transformers 4.45.x.

## 19. Final Submission Checklist

- [x] `README.md` explains the project and final result.
- [x] `docs/PRD.md` is expanded.
- [x] `docs/PLAN.md` is expanded.
- [x] `docs/TODO.md` is expanded.
- [x] `docs/REPORT.md` contains final technical conclusions.
- [x] Specialized PRDs contain sufficient detail.
- [x] `configs/experiment.yaml` points to the final stress experiment.
- [x] Source code remains modular.
- [x] Tests remain lightweight.
- [x] Ruff passes.
- [x] Pytest passes.
- [x] Final figures render.
- [x] Final comparison table is present.
- [x] Heavy artifacts remain ignored.
- [x] Private PDFs remain ignored.
- [x] Git status contains only intended changes.

## 20. File-Level Implementation Trace

- [x] `src/airllm_ex05/__init__.py` exists so the package is importable.
- [x] `src/airllm_ex05/constants.py` centralizes repeated status/runner constants.
- [x] `src/airllm_ex05/config.py` owns YAML config parsing.
- [x] `src/airllm_ex05/config.py` owns typed config validation.
- [x] `src/airllm_ex05/config.py` owns path resolution.
- [x] `src/airllm_ex05/models.py` owns data contracts.
- [x] `src/airllm_ex05/benchmark.py` owns result persistence.
- [x] `src/airllm_ex05/hardware.py` owns machine inspection.
- [x] `src/airllm_ex05/metrics.py` owns timing and memory metric helpers.
- [x] `src/airllm_ex05/cost_analysis.py` owns API/local cost comparison.
- [x] `src/airllm_ex05/plotting.py` owns generated chart creation.
- [x] `src/airllm_ex05/report.py` owns generated report content.
- [x] `src/airllm_ex05/cli.py` owns command-line routing.
- [x] `src/airllm_ex05/runners/__init__.py` keeps runner package imports explicit.
- [x] `src/airllm_ex05/runners/common.py` owns shared runner mechanics.
- [x] `src/airllm_ex05/runners/baseline_runner.py` owns direct Transformers inference.
- [x] `src/airllm_ex05/runners/airllm_runner.py` owns AirLLM inference attempts.
- [x] `src/airllm_ex05/runners/quantized_runner.py` owns quantized inference attempts.
- [x] `src/airllm_ex05/shared/__init__.py` keeps shared utilities importable.
- [x] `src/airllm_ex05/shared/logging_utils.py` owns logging setup.
- [x] `src/airllm_ex05/shared/paths.py` owns directory helpers.
- [x] `src/airllm_ex05/shared/validation.py` owns small validation helpers.
- [x] `tests/test_config.py` covers config behavior.
- [x] `tests/test_hardware.py` covers hardware model behavior.
- [x] `tests/test_metrics.py` covers metric helpers.
- [x] `tests/test_benchmark.py` covers persistence behavior.
- [x] `tests/test_runners.py` covers runner success and failure paths.
- [x] `tests/test_cli.py` covers command wiring.
- [x] `tests/test_cost_analysis.py` covers cost math.
- [x] `tests/test_report.py` covers report/analysis outputs.
- [x] `tests/test_shared_utils.py` covers utility helpers.
- [x] `tests/test_misc_coverage.py` raises coverage over less central branches.
- [x] `configs/experiment.yaml` is the single committed experiment config.
- [x] `results/figures/latency.png` is final visual evidence.
- [x] `results/figures/throughput.png` is final visual evidence.
- [x] `results/figures/memory.png` is final visual evidence.
- [x] `results/figures/cost_curve.png` is final visual evidence.
- [x] `results/processed/comparison_table.csv` is final tabular evidence.
- [x] `results/processed/analysis.json` summarizes final analysis.
- [x] `README.md` is the evaluator-facing overview.

## 21. Exact Evidence Fields To Preserve

- [x] Preserve `runner=baseline` rows.
- [x] Preserve `runner=airllm` rows.
- [x] Preserve `runner=quantized` rows.
- [x] Preserve `status=success` rows.
- [x] Preserve `status=failed` rows when a future run fails.
- [x] Preserve `model_name=Qwen/Qwen2.5-7B-Instruct` in final result files.
- [x] Preserve prompt index 0.
- [x] Preserve prompt index 1.
- [x] Preserve run index 0.
- [x] Preserve the AirLLM raw-string failure as historical diagnosis in docs.
- [x] Preserve final AirLLM success metrics.
- [x] Preserve baseline load-time metric.
- [x] Preserve baseline latency metrics.
- [x] Preserve baseline TTFT metrics.
- [x] Preserve baseline TPOT metrics.
- [x] Preserve baseline throughput metrics.
- [x] Preserve baseline RAM metrics.
- [x] Preserve final quantized success rows.
- [x] Preserve quantized bitsandbytes metadata.
- [x] Preserve quantized latency metrics.
- [x] Preserve quantized TTFT metrics.
- [x] Preserve quantized TPOT metrics.
- [x] Preserve quantized throughput metrics.
- [x] Preserve quantized RAM metrics.
- [x] Preserve `break_even_monthly_requests: null`.
- [x] Preserve cost points for 100 requests.
- [x] Preserve cost points for 1000 requests.
- [x] Preserve cost points for 10000 requests.
- [x] Preserve cost points for 100000 requests.

## 22. Final Documentation Consistency Checks

- [x] Confirm `docs/PLAN.md` and `docs/PRD.md` agree on architecture.
- [x] Confirm `docs/PRD_airllm_pipeline.md` and `docs/REPORT.md` agree on AirLLM status.
- [x] Confirm `docs/PRD_benchmarking.md` and `docs/REPORT.md` agree on result counts.
- [x] Confirm `docs/PRD_quantization.md` and `docs/REPORT.md` agree on bitsandbytes claims.
- [x] Confirm `docs/TODO.md` does not mark human review as completed prematurely.
- [x] Confirm `README.md` and `docs/REPORT.md` both explain the successful final run.
- [x] Confirm all generated figures are referenced from at least one doc.
- [x] Confirm all CLI commands are shown with `uv run`.
- [x] Confirm all paths use repository-relative examples.
- [x] Confirm all final claims can be traced to code, config, or result files.
