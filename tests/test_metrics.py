from airllm_ex05.metrics import bytes_to_gb, count_simple_tokens, time_per_token, tokens_per_second


def test_count_simple_tokens() -> None:
    assert count_simple_tokens("hello local llm") == 3
    assert count_simple_tokens("   ") == 0


def test_throughput_and_tpot() -> None:
    assert tokens_per_second(10, 2.0) == 5.0
    assert time_per_token(10, 2.0) == 0.2
    assert tokens_per_second(0, 2.0) is None
    assert time_per_token(10, 0.0) is None


def test_bytes_to_gb() -> None:
    assert bytes_to_gb(1024**3) == 1.0
    assert bytes_to_gb(None) is None
