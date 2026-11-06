"""HAL environment + GPU detection contracts.

The mutation gate runs mutants over `_detect_environment` and `_check_gpu`;
these tests pin each branch so a dropped rule or a constant return goes red.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.infra.hal import HALBlock


@pytest.fixture
def hal() -> HALBlock:
    return HALBlock()


def test_detect_environment_local_by_default(hal, monkeypatch):
    for var in ("RAILWAY", "AWS_LAMBDA_FUNCTION_NAME", "GCP_PROJECT", "JETSON"):
        monkeypatch.delenv(var, raising=False)
    assert hal._detect_environment() is "local"


@pytest.mark.parametrize("var,expected", [("RAILWAY", "cloud"), ("JETSON", "edge")])
def test_detect_environment_env_switches(hal, monkeypatch, var, expected):
    for v in ("RAILWAY", "AWS_LAMBDA_FUNCTION_NAME", "GCP_PROJECT", "JETSON"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "1")
    assert hal._detect_environment() == expected


def test_check_gpu_true_when_nvidia_smi_succeeds(hal, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=""))
    assert hal._check_gpu() is True


def test_check_gpu_metal_on_darwin(hal, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="Metal: Supported"))
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert hal._check_gpu() is True


def test_check_gpu_false_when_no_gpu(hal, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert hal._check_gpu() is False
