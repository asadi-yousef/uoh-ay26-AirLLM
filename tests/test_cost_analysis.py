from airllm_ex05.config import load_config
from airllm_ex05.cost_analysis import (
    break_even_request_count,
    build_cost_curve,
    estimate_api_cost,
    estimate_local_monthly_cost,
)


def test_cost_estimates_are_positive() -> None:
    config = load_config("configs/experiment.yaml")

    assert estimate_api_cost(100, 50, 1000, config.cost) > 0
    assert estimate_local_monthly_cost(config.cost, runtime_hours=10) > 0


def test_cost_curve_and_break_even() -> None:
    config = load_config("configs/experiment.yaml")

    points = build_cost_curve(1000, 500, 2.0, config.cost)

    assert len(points) == len(config.cost.monthly_request_volumes)
    assert break_even_request_count(points) is None or isinstance(
        break_even_request_count(points), int
    )
