# AirLLM Pipeline PRD

## Objective

Attempt AirLLM-style inference for the same model and prompts as the baseline, while documenting the paging tradeoff described in the lecture.

## Lecture Concepts Used

AirLLM avoids loading all weights into VRAM/RAM simultaneously. It executes layer by layer and uses disk-backed weight movement similar to virtual memory paging. This makes larger models accessible on modest hardware, but disk I/O increases latency and lowers throughput.

## Functional Requirements

- Read model name, cache paths, device preference, and layer shard path from config.
- Import AirLLM lazily so the rest of the project works without it.
- Save a structured failed result if AirLLM is unavailable, incompatible with the model, or lacks required hardware.
- Record AirLLM-specific settings in the result metadata.

## Acceptance Criteria

- Running `airllm-ex05 airllm --config configs/experiment.yaml` never crashes the whole benchmark because of AirLLM import or model errors.
- The report can explain exactly whether AirLLM ran, failed to install, failed to load, or ran slowly.
