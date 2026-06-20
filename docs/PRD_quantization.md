# Quantization PRD

## Objective

Measure whether lower-precision loading reduces memory pressure enough to improve local inference, while documenting quality and compatibility tradeoffs.

## Lecture Concepts Used

Quantization reduces the number of bits used to represent weights. Lower precision can reduce RAM/VRAM and bandwidth requirements, especially during Decode, but aggressive quantization can degrade output quality. QLoRA and NF4 show why 4-bit formats can be useful, although this project focuses on inference.

## Functional Requirements

- Support configurable quantization mode: `8bit`, `4bit`, or fallback dtype loading.
- Use `transformers` quantization APIs when available.
- Treat `bitsandbytes` as optional, especially because Windows support may be limited.
- Save the attempted quantization method and failure details.

## Acceptance Criteria

- Tests do not import or require `bitsandbytes`.
- Quantized failures are saved with method, model, prompt, and error message.
- Analysis compares quantized memory, speed, and output sample with baseline and AirLLM when data exists.
