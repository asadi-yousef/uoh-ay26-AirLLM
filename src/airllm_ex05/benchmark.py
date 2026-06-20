"""Benchmark result serialization."""

import csv
import json
from pathlib import Path
from typing import Any

from airllm_ex05.models import BenchmarkResult, write_json_model


def save_result(result: BenchmarkResult, raw_dir: Path) -> Path:
    """Save one raw benchmark result as JSON."""
    return write_json_model(result, raw_dir / result.filename())


def load_results(raw_dir: Path) -> list[BenchmarkResult]:
    """Load benchmark JSON files from a raw result directory."""
    results = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name == "hardware.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(BenchmarkResult.model_validate(data))
    return results


def save_results_csv(results: list[BenchmarkResult], output_path: Path) -> Path:
    """Save a flat CSV table for spreadsheet inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_result(result) for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _flatten_result(result: BenchmarkResult) -> dict[str, Any]:
    metrics = result.metrics.model_dump()
    return {
        "runner": result.runner,
        "model_name": result.model_name,
        "prompt_index": result.prompt_index,
        "run_index": result.run_index,
        "status": result.status,
        "error_type": result.error_type,
        "error_message": result.error_message,
        **metrics,
    }
