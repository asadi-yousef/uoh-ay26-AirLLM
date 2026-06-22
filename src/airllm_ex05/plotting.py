"""Plot generation for benchmark analysis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from airllm_ex05.cost_analysis import CostPoint
from airllm_ex05.models import BenchmarkResult


def plot_metric(results: list[BenchmarkResult], metric_name: str, output_path: Path) -> Path | None:
    """Create a bar chart for a benchmark metric."""
    grouped: dict[str, list[float]] = {}
    for result in results:
        grouped.setdefault(result.runner, [])
        value = getattr(result.metrics, metric_name)
        if value is not None:
            grouped[result.runner].append(value)
    if not grouped:
        return None
    labels = list(grouped)
    values = [
        sum(metric_values) / len(metric_values) if metric_values else 0.0
        for metric_values in grouped.values()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values)
    for bar, metric_values in zip(bars, grouped.values(), strict=True):
        if metric_values:
            continue
        bar.set_hatch("//")
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            0,
            "N/A",
            ha="center",
            va="bottom",
        )
    plt.ylabel(metric_name.replace("_", " "))
    plt.xlabel("runner")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_cost_curve(points: list[CostPoint], output_path: Path) -> Path | None:
    """Plot cumulative local versus API cost."""
    if not points:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requests = [point.monthly_requests for point in points]
    api = [point.api_cost_usd for point in points]
    local = [point.local_cost_usd for point in points]
    plt.figure(figsize=(8, 4))
    plt.plot(requests, api, marker="o", label="API")
    plt.plot(requests, local, marker="o", label="On-Prem")
    plt.xscale("log")
    plt.xlabel("monthly requests")
    plt.ylabel("monthly cost (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
