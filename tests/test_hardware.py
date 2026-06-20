from pathlib import Path

from airllm_ex05.hardware import collect_hardware_info, save_hardware_info


def test_hardware_info_schema() -> None:
    info = collect_hardware_info()

    assert info.logical_cores >= 1
    assert info.ram_total_gb > 0
    assert info.python_version
    assert isinstance(info.cuda_available, bool)


def test_save_hardware_info(tmp_path: Path) -> None:
    output = save_hardware_info(tmp_path / "hardware.json")

    assert output.exists()
    assert "cpu_model" in output.read_text(encoding="utf-8")
