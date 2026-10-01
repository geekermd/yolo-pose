"""Inference backend discovery and selection.

The NumPy engine always works.  When ONNX Runtime is importable this module works
out which hardware execution providers it can offer — AMD XDNA NPU (Vitis AI),
ROCm/MIGraphX GPU, OpenVINO, DirectML, CUDA — and picks the fastest one that is
actually usable.

Accelerator support needs three things to line up:

1. the kernel driver and device node (e.g. ``/dev/accel/accel0`` for the AMD NPU),
2. permission to open that node (group membership such as ``render``),
3. an ONNX Runtime build that ships a matching execution provider plugin.

:func:`diagnose` reports exactly which of those is missing.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["ProviderInfo", "available_providers", "resolve_providers",
           "describe_providers", "diagnose", "NumpyBackendName"]

NumpyBackendName = "numpy"

# Provider preference: NPU first, then GPU, then generic accelerators.
_PREFERENCE: Tuple[str, ...] = (
    "VitisAIExecutionProvider",      # AMD XDNA NPU (Ryzen AI)
    "RyzenAIExecutionProvider",      # newer AMD plugin EP naming
    "MIGraphXExecutionProvider",     # AMD GPU via MIGraphX
    "ROCMExecutionProvider",         # AMD GPU via ROCm
    "OpenVINOExecutionProvider",     # Intel NPU/GPU/CPU
    "DmlExecutionProvider",          # DirectML (Windows)
    "QNNExecutionProvider",          # Qualcomm Hexagon NPU
    "CoreMLExecutionProvider",       # Apple ANE/GPU
    "TensorrtExecutionProvider",     # NVIDIA
    "CUDAExecutionProvider",         # NVIDIA
    "CPUExecutionProvider",
)

# Human-readable aliases for --provider.
PROVIDER_ALIASES: Dict[str, str] = {
    "auto": "",
    "cpu": "CPUExecutionProvider",
    "numpy": NumpyBackendName,
    "npu": "VitisAIExecutionProvider",
    "xdna": "VitisAIExecutionProvider",
    "vitis": "VitisAIExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "ryzenai": "RyzenAIExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "gpu": "MIGraphXExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorRTExecutionProvider",
}

# Device nodes an accelerator provider needs, with the group that owns them.
_ACCELERATOR_NODES = {
    "VitisAIExecutionProvider": ("/dev/accel/accel0", "render"),
    "RyzenAIExecutionProvider": ("/dev/accel/accel0", "render"),
    "MIGraphXExecutionProvider": ("/dev/kfd", "render"),
    "ROCMExecutionProvider": ("/dev/kfd", "render"),
    "QNNExecutionProvider": ("/dev/fastrpc-cdsp", "adsprpc"),
    "TensorrtExecutionProvider": (None, None),
    "CUDAExecutionProvider": ("/dev/nvidiactl", "video"),
    "DmlExecutionProvider": (None, None),
    "CoreMLExecutionProvider": (None, None),
    "OpenVINOExecutionProvider": (None, None),
    "CPUExecutionProvider": (None, None),
}


@dataclass
class ProviderInfo:
    """One execution provider and whether it can actually be used."""

    name: str
    available: bool
    reason: str = ""
    device: Optional[str] = None
    required_group: Optional[str] = None
    is_accelerator: bool = False

    @property
    def short(self) -> str:
        if self.name == "VitisAIExecutionProvider":
            return "NPU (AMD XDNA / Vitis AI)"
        if self.name == "CPUExecutionProvider":
            return "CPU"
        return self.name.replace("ExecutionProvider", "")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "short": self.short,
            "available": self.available,
            "reason": self.reason,
            "device": self.device,
            "required_group": self.required_group,
            "accelerator": self.is_accelerator,
        }


def _device_ok(path: Optional[str]) -> Tuple[bool, str]:
    """Check that a device node exists and is openable."""
    if not path:
        return True, ""
    if not os.path.exists(path):
        return False, f"{path} does not exist"
    try:
        descriptor = os.open(path, os.O_RDWR)
        os.close(descriptor)
        return True, ""
    except OSError as exc:
        return False, f"{path} is not accessible ({exc})"


def _in_group(group: Optional[str]) -> bool:
    if not group:
        return True
    try:
        import grp

        gids = os.getgroups()
        target = grp.getgrnam(group).gr_gid
        return target in gids or os.getgid() == target
    except Exception:
        return False


def onnxruntime_providers() -> Optional[List[str]]:
    """Execution providers compiled into the installed ONNX Runtime, if any."""
    try:
        import onnxruntime as ort
    except Exception:
        return None
    try:
        return list(ort.get_available_providers())
    except Exception:
        return None


def available_providers() -> List[ProviderInfo]:
    """Every provider we know about, annotated with usability.

    A provider is reported available only when ONNX Runtime exposes it *and* the
    device node it needs can be opened.
    """
    compiled = onnxruntime_providers()
    infos: List[ProviderInfo] = []

    for name in _PREFERENCE:
        device, group = _ACCELERATOR_NODES.get(name, (None, None))
        is_accelerator = name != "CPUExecutionProvider"

        if compiled is None:
            infos.append(ProviderInfo(
                name=name, available=False,
                reason="onnxruntime is not installed",
                device=device, required_group=group,
                is_accelerator=is_accelerator))
            continue

        if name not in compiled:
            if name == "CPUExecutionProvider":
                infos.append(ProviderInfo(
                    name=name, available=False,
                    reason="not reported by onnxruntime"))
            else:
                infos.append(ProviderInfo(
                    name=name, available=False,
                    reason="this onnxruntime build does not include the provider",
                    device=device, required_group=group,
                    is_accelerator=is_accelerator))
            continue

        device_ok, why = _device_ok(device)
        if not device_ok:
            hint = ""
            if group and not _in_group(group):
                hint = f"; add your user to the '{group}' group and re-login"
            infos.append(ProviderInfo(
                name=name, available=False, reason=why + hint,
                device=device, required_group=group,
                is_accelerator=is_accelerator))
            continue

        infos.append(ProviderInfo(
            name=name, available=True, reason="ready",
            device=device, required_group=group,
            is_accelerator=is_accelerator))

    return infos


def resolve_providers(request: str = "auto") -> List[str]:
    """Turn a ``--provider`` value into an ONNX Runtime provider list.

    ``auto`` prefers accelerators in :data:`_PREFERENCE` order.  ``numpy``
    returns an empty list, which tells the caller to use the NumPy engine.
    """
    wanted = (request or "auto").strip()
    lowered = wanted.lower()

    if lowered in ("", "auto"):
        infos = {info.name: info for info in available_providers()}
        chosen = [name for name in _PREFERENCE
                  if name in infos and infos[name].available]
        if not chosen or chosen == ["CPUExecutionProvider"]:
            return chosen or ["CPUExecutionProvider"]
        # Accelerator first, CPU as the fallback that always works.
        if "CPUExecutionProvider" not in chosen and \
                "CPUExecutionProvider" in infos:
            chosen.append("CPUExecutionProvider")
        return chosen

    canonical = PROVIDER_ALIASES.get(lowered)
    if canonical is None:
        # Accept the raw provider name too.
        canonical = wanted if wanted.endswith("ExecutionProvider") else wanted
    if canonical == NumpyBackendName:
        return []
    return [canonical]


def describe_providers() -> str:
    """A short human-readable provider report."""
    compiled = onnxruntime_providers()
    lines = []
    if compiled is None:
        lines.append("onnxruntime : not installed (pure NumPy backend in use)")
    else:
        import onnxruntime as ort
        lines.append(f"onnxruntime : {ort.__version__}")
        lines.append(f"compiled EP : {', '.join(compiled)}")
    lines.append("")
    lines.append(f"{'provider':<32} {'state':<8} note")
    for info in available_providers():
        if not info.available and "not installed" in info.reason:
            continue
        state = "ready" if info.available else "no"
        note = info.reason if not info.available else (
            f"device {info.device}" if info.device else "no device node needed")
        lines.append(f"{info.short:<32} {state:<8} {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# hardware diagnosis
# --------------------------------------------------------------------------- #

def _run(command: Sequence[str], timeout: float = 4.0) -> str:
    if not shutil.which(command[0]):
        return ""
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return (result.stdout or "") + (result.stderr or "")
    except Exception:
        return ""


def diagnose() -> Dict[str, object]:
    """Inspect the machine for NPU/GPU capability and report what is missing.

    Everything here is read-only and needs no elevated privileges.
    """
    report: Dict[str, object] = {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": "",
        "devices": {},
        "drivers": [],
        "firmware": [],
        "onnxruntime": None,
        "providers": [info.as_dict() for info in available_providers()],
        "blockers": [],
        "next_steps": [],
    }

    cpu_info = ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    cpu_info = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    report["cpu"] = cpu_info

    # Device nodes that matter for acceleration.
    for path in ("/dev/accel/accel0", "/dev/dri/renderD128", "/dev/kfd",
                 "/dev/nvidiactl", "/dev/fastrpc-cdsp"):
        if os.path.exists(path):
            accessible = True
            try:
                descriptor = os.open(path, os.O_RDWR)
                os.close(descriptor)
            except OSError:
                accessible = False
            report["devices"][path] = "accessible" if accessible else "permission denied"

    modules = _run(["lsmod"])
    for module in ("amdxdna", "amdgpu", "i915", "xe", "ivpu", "hailo_pci"):
        if module in modules:
            report["drivers"].append(module)

    firmware_dir = "/lib/firmware/amdnpu"
    if os.path.isdir(firmware_dir):
        for entry in sorted(os.listdir(firmware_dir)):
            report["firmware"].append(entry)

    compiled = onnxruntime_providers()
    if compiled is not None:
        import onnxruntime as ort
        report["onnxruntime"] = {"version": ort.__version__, "compiled": compiled}

    # ---- blockers and advice --------------------------------------------- #
    device = report["devices"].get("/dev/accel/accel0")
    if device == "permission denied":
        try:
            import grp
            account_has_group = "render" in list(grp.getgrnam("render").gr_mem)
        except Exception:
            account_has_group = False
        report["blockers"].append(
            "/dev/accel/accel0 is not readable in this session"
            + (" (you are in the 'render' group, but this session predates it)"
               if account_has_group else ""))
        report["next_steps"].append(
            "log out and back in to apply the 'render' group, or use it for one "
            "command:  sg render -c '<command>'")
        if not account_has_group:
            report["next_steps"].append(
                "sudo usermod -aG render $USER")
    if "amdxdna" in report["drivers"] and device is None:
        report["blockers"].append(
            "the amdxdna driver is loaded but /dev/accel/accel0 is missing")
    if compiled is None:
        report["blockers"].append("onnxruntime is not installed")
        report["next_steps"].append(
            "pip install onnxruntime            # CPU only, no NPU support")
        report["next_steps"].append(
            "install the AMD Ryzen AI software stack for NPU execution")
    elif not any("Vitis" in name or "RyzenAI" in name for name in compiled):
        report["blockers"].append(
            "the installed onnxruntime has no AMD NPU execution provider")
        report["next_steps"].append(
            "install the AMD Ryzen AI / Vitis AI execution provider and point "
            "ONNX Runtime at it (see README, 'Hardware acceleration')")
    return report


def format_diagnosis(report: Dict[str, object]) -> str:
    """Render :func:`diagnose` output for the terminal."""
    lines = [
        "hardware acceleration diagnosis",
        f"  platform   : {report.get('platform')}",
        f"  kernel     : {report.get('kernel')}",
        f"  cpu        : {report.get('cpu') or 'unknown'}",
    ]
    drivers = report.get("drivers") or []
    lines.append(f"  drivers    : {', '.join(drivers) if drivers else 'none detected'}")

    devices = report.get("devices") or {}
    if devices:
        lines.append("  devices    :")
        for path, state in devices.items():
            lines.append(f"      {path:<26} {state}")
    else:
        lines.append("  devices    : no accelerator nodes found")

    firmware = report.get("firmware") or []
    if firmware:
        lines.append(f"  npu fw     : {', '.join(firmware[:4])}")

    ort = report.get("onnxruntime")
    if ort:
        lines.append(f"  onnxruntime: {ort['version']}")
        lines.append(f"  compiled EP: {', '.join(ort['compiled'])}")
    else:
        lines.append("  onnxruntime: not installed (NumPy backend only)")

    lines.append("")
    lines.append("  execution providers:")
    for info in report.get("providers") or []:
        if not info["accelerator"] and info["name"] != "CPUExecutionProvider":
            continue
        state = "ready" if info["available"] else "unavailable"
        note = info["reason"] if not info["available"] else (
            info["device"] or "no device needed")
        lines.append(f"      {info['short']:<32} {state:<12} {note}")

    blockers = report.get("blockers") or []
    steps = report.get("next_steps") or []
    if blockers:
        lines.append("")
        lines.append("  blockers:")
        for blocker in blockers:
            lines.append(f"      - {blocker}")
    if steps:
        lines.append("")
        lines.append("  to enable acceleration:")
        for step in steps:
            lines.append(f"      $ {step}")
    return "\n".join(lines)
