"""Economic analysis for local inference versus API usage."""

from pydantic import BaseModel

from airllm_ex05.config import CostConfig


class CostPoint(BaseModel):
    """Cost estimate for one monthly request volume."""

    monthly_requests: int
    api_cost_usd: float
    local_cost_usd: float
    api_cost_per_request_usd: float
    local_cost_per_request_usd: float


def estimate_api_cost(
    input_tokens: int,
    output_tokens: int,
    requests: int,
    cost: CostConfig,
    cached_input_ratio: float = 0.0,
) -> float:
    """Estimate API cost with optional prompt-cache discount."""
    cached_tokens = input_tokens * cached_input_ratio
    normal_input_tokens = input_tokens - cached_tokens
    input_cost = normal_input_tokens * cost.api_input_price_per_1m_tokens_usd / 1_000_000
    cached_cost = (
        cached_tokens
        * cost.api_input_price_per_1m_tokens_usd
        * cost.cached_input_discount
        / 1_000_000
    )
    output_cost = output_tokens * cost.api_output_price_per_1m_tokens_usd / 1_000_000
    return round((input_cost + cached_cost + output_cost) * requests, 6)


def estimate_local_monthly_cost(cost: CostConfig, runtime_hours: float) -> float:
    """Estimate monthly local cost from amortized hardware, power, and maintenance."""
    capex = cost.hardware_cost_usd / cost.hardware_lifetime_months
    electricity = (cost.average_power_watts / 1000) * runtime_hours * cost.electricity_usd_per_kwh
    return round(capex + electricity + cost.maintenance_monthly_usd, 6)


def build_cost_curve(
    input_tokens: int,
    output_tokens: int,
    seconds_per_request: float,
    cost: CostConfig,
) -> list[CostPoint]:
    """Build cost comparison points for configured request volumes."""
    points = []
    for requests in cost.monthly_request_volumes:
        runtime_hours = requests * seconds_per_request / 3600
        api_cost = estimate_api_cost(input_tokens, output_tokens, requests, cost, 0.5)
        local_cost = estimate_local_monthly_cost(cost, runtime_hours)
        points.append(
            CostPoint(
                monthly_requests=requests,
                api_cost_usd=api_cost,
                local_cost_usd=local_cost,
                api_cost_per_request_usd=round(api_cost / requests, 8),
                local_cost_per_request_usd=round(local_cost / requests, 8),
            )
        )
    return points


def break_even_request_count(points: list[CostPoint]) -> int | None:
    """Return the first request volume where local cost is cheaper than API."""
    for point in points:
        if point.local_cost_usd <= point.api_cost_usd:
            return point.monthly_requests
    return None
