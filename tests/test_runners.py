from airllm_ex05.config import load_config
from airllm_ex05.runners.airllm_runner import run_airllm
from airllm_ex05.runners.baseline_runner import run_baseline
from airllm_ex05.runners.quantized_runner import run_quantized


class FakeTensor:
    def to(self, _device: str) -> "FakeTensor":
        return self


class FakeTokenizer:
    def __call__(self, _prompt: str, return_tensors: str) -> dict[str, FakeTensor]:
        assert return_tensors == "pt"
        return {"input_ids": FakeTensor()}

    def decode(self, _output: object, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "generated text"


class FakeModel:
    device = "cpu"

    def eval(self) -> None:
        return None

    def generate(self, **_kwargs: object) -> list[list[int]]:
        return [[1, 2, 3]]


class FakeTransformers:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> FakeTokenizer:
            return FakeTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> FakeModel:
            return FakeModel()

    class BitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs


class FakeTorch:
    float16 = "float16"
    float32 = "float32"
    qint8 = "qint8"

    class Nn:
        class Linear:
            pass

    nn = Nn

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    cuda = Cuda

    class Ao:
        class Quantization:
            @staticmethod
            def quantize_dynamic(model: object, layers: set[object], dtype: str) -> object:
                assert FakeTorch.nn.Linear in layers
                assert dtype == FakeTorch.qint8
                return model

        quantization = Quantization

    ao = Ao


class FakeAirLLMModel:
    @staticmethod
    def from_pretrained(*_args: object, **_kwargs: object) -> "FakeAirLLMModel":
        return FakeAirLLMModel()

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return f"{prompt} {max_new_tokens}"


class FakeAirLLM:
    AutoModel = FakeAirLLMModel


def _fake_import(name: str) -> object:
    if name == "transformers":
        return FakeTransformers
    if name == "torch":
        return FakeTorch
    if name == "airllm":
        return FakeAirLLM
    msg = name
    raise ImportError(msg)


def _missing_transformers_import(name: str) -> object:
    if name in {"transformers", "torch"}:
        raise ImportError(name)
    return _fake_import(name)


def test_baseline_missing_transformers_returns_failures(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", _missing_transformers_import)
    config = load_config("configs/experiment.yaml")
    results = run_baseline(config)

    assert results
    assert all(result.status == "failed" for result in results)
    assert all(result.metadata["stage"] == "load" for result in results)


def test_baseline_success_with_fakes(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", _fake_import)
    config = load_config("configs/experiment.yaml")

    results = run_baseline(config)

    assert all(result.status == "success" for result in results)
    assert results[0].generated_text == "generated text"


def test_airllm_missing_dependency_returns_failures() -> None:
    config = load_config("configs/experiment.yaml")
    results = run_airllm(config)

    assert results
    assert all(result.runner == "airllm" for result in results)
    assert all(result.status == "failed" for result in results)


def test_airllm_success_with_fakes(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", _fake_import)
    config = load_config("configs/experiment.yaml")

    results = run_airllm(config)

    assert all(result.status == "success" for result in results)
    assert "layer_shards" in results[0].metadata


def test_quantized_missing_transformers_returns_failures(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", _missing_transformers_import)
    config = load_config("configs/experiment.yaml")
    results = run_quantized(config)

    assert results
    assert all(result.runner == "quantized" for result in results)
    assert all(result.status == "failed" for result in results)


def test_quantized_success_with_fakes(monkeypatch) -> None:
    monkeypatch.setattr("importlib.import_module", _fake_import)
    config = load_config("configs/experiment.yaml")

    results = run_quantized(config)

    assert all(result.status == "success" for result in results)
    assert results[0].metadata["bits"] == 8
    assert results[0].metadata["quantization_method"] == "torch.dynamic_int8"
