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
        value = getattr(result.metrics, metric_name)
        if value is not None:
            grouped.setdefault(result.runner, [])
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
    plt.bar(labels, values)
    plt.ylabel(metric_name.replace("_", " "))
    plt.xlabel("runner")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_runner_outcomes(results: list[BenchmarkResult], output_path: Path) -> Path | None:
    """Create a stacked bar chart showing success and failure counts by runner."""
    if not results:
        return None
    runners = sorted({result.runner for result in results})
    success_counts = [
        sum(result.runner == runner and result.status == "success" for result in results)
        for runner in runners
    ]
    failure_counts = [
        sum(result.runner == runner and result.status == "failed" for result in results)
        for runner in runners
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(runners, success_counts, label="success")
    plt.bar(runners, failure_counts, bottom=success_counts, label="failed")
    plt.ylabel("result rows")
    plt.xlabel("runner")
    plt.legend()
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
