"""High-level pose detection: preprocessing, inference, decoding, annotation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.runtime import NumpyEngine, OnnxRuntimeError
from .imaging import LetterboxInfo, letterbox, load_image, to_network_input
from .joints import JointAngles, compute_joint_angles
from .pose import KEYPOINT_NAMES, Pose, decode_poses

__all__ = ["Detector", "DetectionResult", "available_backends"]


def _default_threads(limit: int = 8) -> int:
    """Thread count for ONNX Runtime: enough to saturate, not to thrash.

    Benchmarks on this project's reference machine (20 threads) showed the knee
    at 4-8 threads; beyond that the extra threads add contention without helping
    a single-frame inference.
    """
    try:
        available = os.cpu_count() or 0
    except Exception:
        available = 0
    if available <= 0:
        return 0
    return max(1, min(limit, available))


def available_backends() -> Dict[str, bool]:
    """Report which inference backends this installation can use."""
    backends = {"numpy": True}
    try:
        import onnxruntime  # noqa: F401
        backends["onnxruntime"] = True
    except Exception:
        backends["onnxruntime"] = False
    return backends


class _OnnxRuntimeEngine:
    """Thin adapter exposing the same ``run`` contract as :class:`NumpyEngine`.

    ``provider`` selects the execution provider: ``auto`` prefers an accelerator
    (AMD XDNA NPU via Vitis AI, then GPU providers) and falls back to the CPU.
    """

    def __init__(self, model_path: str, threads: int = 0,
                 provider: str = "auto"):
        import onnxruntime as ort

        from ..core.accelerators import resolve_providers

        options = ort.SessionOptions()
        # ONNX Runtime's own default uses very few cores on a big CPU and leaves
        # a lot of throughput on the table: on a 20-thread machine, letting it
        # choose took 53 ms per frame versus 26 ms with 4 threads and 24 ms with
        # 8.  Cap the default at 8 (the measured knee) unless the caller asks for
        # something specific.
        resolved_threads = int(threads) if threads else _default_threads()
        if resolved_threads:
            options.intra_op_num_threads = resolved_threads
            options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        requested = resolve_providers(provider)
        if not requested:
            requested = ["CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(
                model_path, options, providers=requested)
        except Exception as exc:
            if provider.lower() not in ("auto", ""):
                # An explicitly requested provider must fail loudly rather than
                # silently dropping back to the CPU.
                raise RuntimeError(
                    f"could not create an ONNX Runtime session with "
                    f"provider(s) {requested}: {exc}") from exc
            self.session = ort.InferenceSession(
                model_path, options, providers=["CPUExecutionProvider"])

        # Record what the session actually got; ONNX Runtime may drop providers.
        try:
            self.active_providers = list(self.session.get_providers())
        except Exception:
            self.active_providers = list(requested)
        self.model = type("M", (), {
            "inputs": [type("V", (), {"name": self.session.get_inputs()[0].name})()],
            "outputs": [type("V", (), {"name": o.name}) for o in self.session.get_outputs()],
        })()
        self.last_timing: Dict[str, float] = {}

    @property
    def input_name(self) -> str:
        return self.session.get_inputs()[0].name

    @property
    def input_shape(self):
        """Static NCHW input shape, or ``None`` when the axes are dynamic."""
        try:
            shape = list(self.session.get_inputs()[0].shape)
        except Exception:
            return None
        normalised = []
        for dim in shape:
            if isinstance(dim, int):
                normalised.append(dim)
            elif isinstance(dim, str) and dim.isdigit():
                normalised.append(int(dim))
            else:
                normalised.append(None)
        return tuple(normalised) if normalised else None

    def run(self, feeds, profile: bool = False):
        return {o.name: v for o, v in zip(self.session.get_outputs(),
                                          self.session.run(None, feeds))}


@dataclass
class DetectionResult:
    """Everything produced for one image."""

    poses: List[Pose]
    joints: List[JointAngles]
    info: LetterboxInfo
    preprocess_seconds: float = 0.0
    inference_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    backend: str = "numpy"

    @property
    def total_seconds(self) -> float:
        return self.preprocess_seconds + self.inference_seconds + self.postprocess_seconds

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "timing": {
                "preprocess_ms": round(self.preprocess_seconds * 1000, 2),
                "inference_ms": round(self.inference_seconds * 1000, 2),
                "postprocess_ms": round(self.postprocess_seconds * 1000, 2),
            },
            "count": len(self.poses),
            "persons": [
                {**pose.as_dict(), "joint_angles": joints.as_dict()}
                for pose, joints in zip(self.poses, self.joints)
            ],
        }


class Detector:
    """YOLO-pose detector with an interchangeable inference backend.

    Parameters
    ----------
    model_path:
        Path to a YOLOv8/YOLO11 pose ``.onnx`` file.
    backend:
        ``"auto"``, ``"numpy"`` or ``"onnxruntime"``.  ``auto`` prefers
        onnxruntime for speed and falls back to the pure-NumPy engine.
    image_size:
        Network input size; the model's static input shape wins when present.
    """

    def __init__(self, model_path: str, backend: str = "auto",
                 image_size: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 max_detections: int = 100, threads: int = 0,
                 fuse: bool = True, verbose: bool = False,
                 provider: str = "auto"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model not found: {model_path}\n"
                "Run `yolo-pose models pull` to download the default model."
            )
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.verbose = verbose
        self.requested_backend = backend
        self.image_size = tuple(int(v) for v in image_size)
        self.provider = provider

        self.backend_name = ""
        self.engine = self._build_engine(backend, threads, fuse, provider)

        # The model's own static input shape always wins: a converted model
        # (e.g. 320x320) must not be fed 640x640 frames.
        shape = getattr(self.engine, "input_shape", None)
        if shape and len(shape) == 4:
            height, width = shape[2], shape[3]
            if isinstance(height, int) and isinstance(width, int) \
                    and height > 0 and width > 0:
                self.image_size = (int(width), int(height))
            else:
                # Dynamic spatial axes: fall back to what the caller asked for.
                self.image_size = tuple(int(v) for v in image_size)
        else:
            self.image_size = tuple(int(v) for v in image_size)

        self.num_keypoints = 17

    # -- construction ------------------------------------------------------- #
    def _build_engine(self, backend: str, threads: int, fuse: bool,
                      provider: str = "auto"):
        wanted = backend.lower()
        if wanted not in ("auto", "numpy", "onnxruntime", "ort"):
            raise ValueError(f"unknown backend {backend!r}")

        if wanted in ("auto", "onnxruntime", "ort"):
            try:
                engine = _OnnxRuntimeEngine(self.model_path, threads, provider)
                self.backend_name = "onnxruntime"
                self.active_providers = getattr(engine, "active_providers", [])
                return engine
            except Exception as exc:
                if wanted != "auto":
                    raise
                if self.verbose:
                    print(f"[yolo-pose] onnxruntime unavailable ({exc}); "
                          "using the pure-NumPy engine")
        engine = NumpyEngine(self.model_path, fuse=fuse, verbose=self.verbose)
        self.backend_name = "numpy"
        self.active_providers = ["NumPyEngine"]
        return engine

    # -- inference ---------------------------------------------------------- #
    def detect(self, image: np.ndarray, keypoint_threshold: float = 0.5
               ) -> DetectionResult:
        """Run detection on an RGB ``uint8`` image array."""
        start = time.perf_counter()
        padded, info = letterbox(image, self.image_size)
        tensor = to_network_input(padded)
        preprocess = time.perf_counter() - start

        start = time.perf_counter()
        outputs = self.engine.run({self.engine.input_name: tensor})
        inference = time.perf_counter() - start

        start = time.perf_counter()
        key = "output0" if "output0" in outputs else next(iter(outputs))
        poses = decode_poses(outputs[key], info,
                             conf_threshold=self.conf_threshold,
                             iou_threshold=self.iou_threshold,
                             max_detections=self.max_detections,
                             max_keypoints=self.num_keypoints)
        joints = [compute_joint_angles(p, keypoint_threshold) for p in poses]
        postprocess = time.perf_counter() - start

        return DetectionResult(poses=poses, joints=joints, info=info,
                               preprocess_seconds=preprocess,
                               inference_seconds=inference,
                               postprocess_seconds=postprocess,
                               backend=self.backend_name)

    def detect_file(self, path: str, keypoint_threshold: float = 0.5
                    ) -> Tuple[np.ndarray, DetectionResult]:
        """Load an image from disk and detect poses in it."""
        image = load_image(path)
        return image, self.detect(image, keypoint_threshold)

    # -- introspection ------------------------------------------------------ #
    def describe(self) -> str:
        lines = [
            f"model      : {os.path.basename(self.model_path)}",
            f"backend    : {self.backend_name}",
            f"providers  : {', '.join(getattr(self, 'active_providers', [])) or 'n/a'}",
            f"input size : {self.image_size[0]}x{self.image_size[1]}",
            f"conf / iou : {self.conf_threshold} / {self.iou_threshold}",
        ]
        if hasattr(self.engine, "describe"):
            lines.append(self.engine.describe())
        else:
            inputs = [i.name for i in self.engine.model.inputs]
            outputs = [o.name for o in self.engine.model.outputs]
            lines.append(f"inputs     : {inputs}")
            lines.append(f"outputs    : {outputs}")
        return "\n".join(lines)
