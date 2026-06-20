# Exercise 05 Technical Report

This report is generated and updated by:

```bash
uv run airllm-ex05 report --config configs/experiment.yaml
```

The project is prepared to document baseline, AirLLM, and quantized local inference experiments. Run the hardware and benchmark commands first, then regenerate this report so the tables, plots, negative results, and economic analysis reflect the local machine.

Key required discussion points are built into the generator: hardware specification, selected model justification, baseline failures, AirLLM paging behavior, quantization memory/speed/quality tradeoffs, Prefill versus Decode, TTFT, TPOT, throughput, latency, VRAM/RAM bottlenecks, on-premises versus API economics, limitations, and final engineering conclusions.
