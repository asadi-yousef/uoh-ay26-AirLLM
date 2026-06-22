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

    def generate(
        self, input_ids: FakeTensor, max_new_tokens: int, **_kwargs: object
    ) -> list[list[int]]:
        assert isinstance(input_ids, FakeTensor)
        assert max_new_tokens > 0
        return [[1, 2, 3]]


class FakeAirLLM:
    AutoModel = FakeAirLLMModel


def fake_import(name: str) -> object:
    if name == "transformers":
        return FakeTransformers
    if name == "torch":
        return FakeTorch
    if name == "airllm":
        return FakeAirLLM
    msg = name
    raise ImportError(msg)


def missing_transformers_import(name: str) -> object:
    if name in {"transformers", "torch"}:
        raise ImportError(name)
    return fake_import(name)
