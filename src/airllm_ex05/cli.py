"""Command line interface for Exercise 05."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from airllm_ex05.benchmark import save_result, save_results_csv
from airllm_ex05.config import load_config
from airllm_ex05.constants import DEFAULT_CONFIG_PATH, HARDWARE_FILENAME
from airllm_ex05.hardware import save_hardware_info
from airllm_ex05.models import BenchmarkResult
from airllm_ex05.runners.airllm_runner import run_airllm
from airllm_ex05.runners.baseline_runner import run_baseline
from airllm_ex05.runners.quantized_runner import run_quantized
from airllm_ex05.shared.logging_utils import configure_logging

Runner = Callable[..., list[BenchmarkResult]]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    return int(args.handler(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airllm-ex05")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(required=True)
    _add_config_command(subparsers, "hardware", _hardware)
    _add_config_command(subparsers, "baseline", _baseline)
    _add_config_command(subparsers, "airllm", _airllm)
    _add_config_command(subparsers, "quantized", _quantized)
    _add_config_command(subparsers, "analyze", _analyze)
    _add_config_command(subparsers, "report", _report)
    return parser


def _add_config_command(
    subparsers: object,
    name: str,
    handler: Callable[[argparse.Namespace], int],
) -> None:
    command = subparsers.add_parser(name)
    command.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to experiment YAML.")
    command.set_defaults(handler=handler)


def _hardware(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = save_hardware_info(config.outputs.raw_dir / HARDWARE_FILENAME)
    print(f"Saved hardware info to {output}")
    return 0


def _baseline(args: argparse.Namespace) -> int:
    return _run_and_save(args.config, run_baseline, "baseline")


def _airllm(args: argparse.Namespace) -> int:
    return _run_and_save(args.config, run_airllm, "airllm")


def _quantized(args: argparse.Namespace) -> int:
    return _run_and_save(args.config, run_quantized, "quantized")


def _run_and_save(config_path: str | Path, runner: Runner, label: str) -> int:
    config = load_config(config_path)
    _clear_runner_outputs(config.outputs.raw_dir, label)
    results = runner(config)
    for result in results:
        save_result(result, config.outputs.raw_dir)
    csv_path = config.outputs.raw_dir / f"{label}_results.csv"
    save_results_csv(results, csv_path)
    successes = sum(result.status == "success" for result in results)
    print(
        f"Saved {len(results)} {label} results "
        f"({successes} successful) to {config.outputs.raw_dir}"
    )
    return 0


def _clear_runner_outputs(raw_dir: Path, label: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in raw_dir.glob(f"{label}_p*_r*.json"):
        path.unlink()
    csv_path = raw_dir / f"{label}_results.csv"
    if csv_path.exists():
        csv_path.unlink()


def _analyze(args: argparse.Namespace) -> int:
    from airllm_ex05.report import analyze_results

    config = load_config(args.config)
    output = analyze_results(config)
    print(f"Saved analysis to {output}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from airllm_ex05.report import generate_report

    config = load_config(args.config)
    output = generate_report(config)
    print(f"Saved report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
