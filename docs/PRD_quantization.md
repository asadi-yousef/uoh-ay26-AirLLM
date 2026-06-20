# Quantization PRD

## Purpose

Attempt lower-precision model loading and inference so the project can compare memory,
latency, throughput, and output quality against direct baseline and AirLLM execution.

## Context

The lecture describes quantization as reducing the number of bits used to represent model
weights. This can reduce RAM/VRAM footprint and memory bandwidth demand, especially during
Decode. The tradeoff is that lower precision can reduce output quality, and backend support
is platform-dependent.

## Relation to Exercise 05

Exercise 05 requires adding quantization to the local inference pipeline and analyzing how it
changes memory, speed, and output quality. The report must identify whether quantization helps
and where the quality red line appears.

## Functional Requirements

- Read quantization mode, bit width, and compute dtype from config.
- Support 4-bit and 8-bit settings.
- Support CPU dynamic-int8 quantization for Windows CPU-only environments.
- Use `transformers.BitsAndBytesConfig` when configured for `bitsandbytes`.
- Use `torch.ao.quantization.quantize_dynamic` when configured for `dynamic_int8`.
- Load tokenizer and model through `AutoTokenizer` and `AutoModelForCausalLM`.
- Return structured failures when `bitsandbytes`, CUDA, or model support is missing.
- Include attempted quantization method and bit width in result metadata on success.

## Non-Functional Requirements

- Tests must not import or require `bitsandbytes`.
- Quantization failure must not abort the whole benchmark.
- Windows and CPU-only limitations must be documented explicitly.
- The report must not claim memory/speed/quality improvements unless result files show them.

## Inputs and Outputs

Inputs:

- `ExperimentConfig.model`
- `ExperimentConfig.quantization`
- Prompt list and benchmark settings
- Installed `transformers`, torch, and optional `bitsandbytes`

Outputs:

- `results/raw/quantized_p*_r*.json`
- `results/raw/quantized_results.csv`
- Processed comparison rows and plots when metrics exist

## Acceptance Criteria

- `uv run airllm-ex05 quantized --config configs/experiment.yaml` writes one result per
  prompt/run.
- Missing or unsupported `bitsandbytes` produces failed results with clear error messages when
  that mode is selected.
- CPU dynamic-int8 runs successfully on the validation machine.
- Successful quantized runs include latency, TPOT, throughput, RAM, VRAM, generated text, and
  quantization metadata.

## Risks and Failure Modes

- Previous observed failure: 4-bit loading required `bitsandbytes>=0.46.1`.
- `bitsandbytes` support may be limited on Windows or CPU-only systems.
- Quantized loading may require CUDA even when baseline CPU loading works.
- Very aggressive quantization may reduce quality.
- Some model architectures may be incompatible with a chosen quantization path.

## Testing Strategy

Tests use fake `transformers`, fake torch, and fake `BitsAndBytesConfig` objects. They verify
successful fake quantized execution and missing-dependency failure behavior without installing
`bitsandbytes`.

## Implementation Notes Connected to Code

- Implemented in `src/airllm_ex05/runners/quantized_runner.py`.
- Reuses baseline `_generate()` for text generation.
- `_quantization_kwargs()` chooses 4-bit or 8-bit `BitsAndBytesConfig` when available, or a
  dtype fallback if the API is absent.
- Current real validation run succeeded for both prompts with `torch.dynamic_int8`.
