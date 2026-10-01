"""Accelerator detection tests (no hardware required)."""

import numpy as np
import pytest

from yolopose.core import accelerators as A


def test_provider_preference_puts_npu_first():
    assert A._PREFERENCE[0] in ("VitisAIExecutionProvider",
                                "RyzenAIExecutionProvider")
    assert A._PREFERENCE[-1] == "CPUExecutionProvider"


def test_resolve_cpu_and_numpy():
    assert A.resolve_providers("cpu") == ["CPUExecutionProvider"]
    assert A.resolve_providers("numpy") == []


def test_resolve_aliases():
    assert A.resolve_providers("npu") == ["VitisAIExecutionProvider"]
    assert A.resolve_providers("xdna") == ["VitisAIExecutionProvider"]
    assert A.resolve_providers("rocm") == ["ROCMExecutionProvider"]
    assert A.resolve_providers("openvino") == ["OpenVINOExecutionProvider"]


def test_resolve_auto_always_includes_a_usable_provider():
    providers = A.resolve_providers("auto")
    assert providers, "auto must never return an empty list"
    assert "CPUExecutionProvider" in providers


def test_known_provider_names_have_device_metadata():
    for name in A._PREFERENCE:
        assert name in A._ACCELERATOR_NODES or name == "CPUExecutionProvider"


def test_provider_info_short_names():
    assert A.ProviderInfo("VitisAIExecutionProvider", True).short.startswith("NPU")
    assert A.ProviderInfo("CPUExecutionProvider", True).short == "CPU"
    assert A.ProviderInfo("ROCMExecutionProvider", True).short == "ROCM"


def test_diagnose_reports_expected_keys():
    report = A.diagnose()
    for key in ("platform", "kernel", "cpu", "devices", "drivers",
                "onnxruntime", "providers", "blockers", "next_steps"):
        assert key in report, key
    text = A.format_diagnosis(report)
    assert "execution providers" in text
    assert "CPU" in text


def test_available_providers_shape():
    infos = A.available_providers()
    assert infos
    names = {info.name for info in infos}
    assert "CPUExecutionProvider" in names
    for info in infos:
        assert isinstance(info.available, bool)
        if not info.available:
            assert info.reason, f"{info.name} unavailable without a reason"


def test_diagnosis_mentions_group_fix_when_device_blocked():
    """If the NPU node exists but is unreadable, the advice must say how to fix it."""
    import os

    if not os.path.exists("/dev/accel/accel0"):
        pytest.skip("no NPU device node on this machine")
    report = A.diagnose()
    blocked = report["devices"].get("/dev/accel/accel0") == "permission denied"
    if blocked:
        assert any("render" in step for step in report["next_steps"])
