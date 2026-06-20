"""Metric calculation and memory sampling helpers."""

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time

import psutil


def count_simple_tokens(text: str) -> int:
    """Approximate token count without requiring a tokenizer."""
    return len(text.split()) if text.strip() else 0


def tokens_per_second(output_tokens: int, latency_seconds: float | None) -> float | None:
    """Compute throughput from output tokens and latency."""
    if not latency_seconds or latency_seconds <= 0 or output_tokens <= 0:
        return None
    return output_tokens / latency_seconds


def time_per_token(output_tokens: int, latency_seconds: float | None) -> float | None:
    """Compute TPOT from output tokens and latency."""
    if not latency_seconds or latency_seconds <= 0 or output_tokens <= 0:
        return None
    return latency_seconds / output_tokens


def bytes_to_gb(value: int | float | None) -> float | None:
    """Convert bytes to GiB rounded for reporting."""
    if value is None:
        return None
    return round(float(value) / (1024**3), 2)


def bytes_to_mb(value: int | float | None) -> float | None:
    """Convert bytes to MiB rounded for reporting."""
    if value is None:
        return None
    return round(float(value) / (1024**2), 2)


def current_process_ram_mb() -> float:
    """Return current process RSS in MiB."""
    return bytes_to_mb(psutil.Process().memory_info().rss) or 0.0


def cuda_peak_memory_mb() -> float | None:
    """Return CUDA peak allocated memory in MiB when torch/CUDA is available."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return bytes_to_mb(torch.cuda.max_memory_allocated())


@dataclass
class MemorySampler:
    """Sample process RAM periodically while work executes."""

    interval_seconds: float = 0.05
    peak_ram_mb: float = 0.0

    def run(self, function: Callable[[], str]) -> str:
        """Run a function while tracking peak process RAM."""
        stop_event = threading.Event()
        thread = threading.Thread(target=self._sample_until, args=(stop_event,), daemon=True)
        thread.start()
        try:
            return function()
        finally:
            stop_event.set()
            thread.join(timeout=1)

    def _sample_until(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            self.peak_ram_mb = max(self.peak_ram_mb, current_process_ram_mb())
            time.sleep(self.interval_seconds)
