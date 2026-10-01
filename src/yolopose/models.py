"""Model discovery and downloading.

Default search order:

1. ``$YOLO_POSE_MODEL`` — explicit override.
2. ``./models/`` relative to the working directory (development checkout).
3. ``/usr/share/yolo-pose/models/`` — where the Debian package installs it.
4. ``~/.local/share/yolo-pose/models/`` — user downloads.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = ["MODELS", "FACE_MODELS", "ModelSpec", "find_model", "find_face_model",
           "search_paths", "pull_model", "default_model_path"]

# Hugging Face mirror first: reachable from networks where huggingface.co is not.
_MIRRORS = (
    "https://hf-mirror.com/{repo}/resolve/main/{file}",
    "https://huggingface.co/{repo}/resolve/main/{file}",
)


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable pose model."""

    name: str
    filename: str
    repo: str
    file: str
    size_mb: float
    description: str
    license: str = "AGPL-3.0"

    def urls(self) -> List[str]:
        return [template.format(repo=self.repo, file=self.file) for template in _MIRRORS]


MODELS: Dict[str, ModelSpec] = {
    "yolov8n-pose": ModelSpec(
        name="yolov8n-pose",
        filename="yolov8n-pose.onnx",
        repo="Xenova/yolov8n-pose",
        file="onnx/model.onnx",
        size_mb=13.5,
        description="YOLOv8-nano pose, COCO-17 keypoints, 640x640 (default; fastest)",
    ),
    "yolov8s-pose": ModelSpec(
        name="yolov8s-pose",
        filename="yolov8s-pose.onnx",
        repo="Xenova/yolov8-pose-onnx",
        file="yolov8s-pose.onnx",
        size_mb=23.0,
        description="YOLOv8-small pose, COCO-17 keypoints (more accurate, slower)",
    ),
    "yolov8m-pose": ModelSpec(
        name="yolov8m-pose",
        filename="yolov8m-pose.onnx",
        repo="Xenova/yolov8-pose-onnx",
        file="yolov8m-pose.onnx",
        size_mb=53.0,
        description="YOLOv8-medium pose, COCO-17 keypoints (accurate, slow on CPU)",
    ),
}

DEFAULT_MODEL = "yolov8n-pose"

# Face pipeline: an SCRFD detector plus an ArcFace/MobileFaceNet embedder.
FACE_MODELS: Dict[str, ModelSpec] = {
    "scrfd-500m": ModelSpec(
        name="scrfd-500m",
        filename="scrfd_500m.onnx",
        repo="deepghs/insightface",
        file="buffalo_s/det_500m.onnx",
        size_mb=2.5,
        description="SCRFD-500M face detector with 5 landmarks (fast)",
    ),
    "scrfd-2.5g": ModelSpec(
        name="scrfd-2.5g",
        filename="scrfd_2.5g.onnx",
        repo="RuteNL/SCRFD-face-detection-ONNX",
        file="2.5g_bnkps.onnx",
        size_mb=3.3,
        description="SCRFD-2.5G face detector with 5 landmarks (more accurate)",
    ),
    "mobilefacenet": ModelSpec(
        name="mobilefacenet",
        filename="w600k_mbf.onnx",
        repo="deepghs/insightface",
        file="buffalo_s/w600k_mbf.onnx",
        size_mb=13.6,
        description="MobileFaceNet (w600k_mbf) 512-d face embedding (fast)",
    ),
}

FACE_DETECTOR = "scrfd-500m"
FACE_EMBEDDER = "mobilefacenet"


def search_paths() -> List[str]:
    """Directories searched for model files, in priority order.

    ``YOLO_POSE_DATA_DIR`` relocates the whole search root, which is useful for
    packaging, containers and tests that stage the payload somewhere else.
    """
    paths: List[str] = []
    override = os.environ.get("YOLO_POSE_MODEL")
    if override:
        paths.append(os.path.dirname(os.path.abspath(override)) or ".")
    paths.append(os.path.join(os.getcwd(), "models"))

    data_root = os.environ.get("YOLO_POSE_DATA_DIR")
    if data_root:
        paths.append(os.path.join(data_root, "share", "yolo-pose", "models"))
        paths.append(os.path.join(data_root, "models"))
        paths.append(data_root)

    paths.append("/usr/share/yolo-pose/models")
    paths.append("/usr/local/share/yolo-pose/models")
    paths.append(os.path.join(os.path.expanduser("~"), ".local", "share",
                              "yolo-pose", "models"))
    # Allow running straight from a source checkout.
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    paths.append(os.path.join(here, "models"))
    return [p for p in dict.fromkeys(paths) if p]


def find_model(name: Optional[str] = None) -> Optional[str]:
    """Return the path of a usable model, or ``None``."""
    override = os.environ.get("YOLO_POSE_MODEL")
    if override and os.path.exists(override):
        return os.path.abspath(override)

    if name:
        if os.path.exists(name):
            return os.path.abspath(name)
        specs = _all_specs()
        spec = specs.get(name) or specs.get(os.path.splitext(name)[0])
        candidates = [spec.filename] if spec else [name]
        if not os.path.splitext(name)[1]:
            candidates.append(name + ".onnx")
    else:
        spec = MODELS[DEFAULT_MODEL]
        candidates = [spec.filename] + [m.filename for m in _all_specs().values()]

    for directory in search_paths():
        for candidate in dict.fromkeys(candidates):
            path = os.path.join(directory, candidate)
            if os.path.exists(path):
                return os.path.abspath(path)
    return None


def _all_specs() -> Dict[str, "ModelSpec"]:
    return {**MODELS, **FACE_MODELS}


def find_face_model(kind: str, name: Optional[str] = None) -> Optional[str]:
    """Locate a face ``detector`` or ``embedder`` model.

    ``kind`` selects the default when ``name`` is omitted.  An explicit ``name``
    may be a path or a registry key.
    """
    if name and os.path.exists(name):
        return os.path.abspath(name)
    if name:
        return find_model(name)
    key = FACE_DETECTOR if kind == "detector" else FACE_EMBEDDER
    found = find_model(key)
    if found:
        return found
    # Fall back to any installed face model of the right family.
    for candidate, spec in FACE_MODELS.items():
        if kind == "detector" and "scrfd" not in candidate:
            continue
        if kind == "embedder" and "scrfd" in candidate:
            continue
        found = find_model(candidate)
        if found:
            return found
    return None


def default_model_path(name: Optional[str] = None) -> str:
    """Path where :func:`pull_model` should store a model."""
    spec = _all_specs().get(name or DEFAULT_MODEL)
    filename = spec.filename if spec else (name or DEFAULT_MODEL)
    target_dir = os.path.join(os.path.expanduser("~"), ".local", "share",
                              "yolo-pose", "models")
    return os.path.join(target_dir, filename)


def _download(url: str, destination: str, timeout: float = 60.0,
              progress: bool = True) -> Tuple[int, str]:
    """Download ``url`` to ``destination``; returns (bytes, sha256)."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "yolo-pose/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        temporary = destination + ".part"
        with open(temporary, "wb") as handle:
            while True:
                chunk = response.read(262144)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if progress and total and sys.stderr.isatty():
                    percent = written * 100.0 / total
                    sys.stderr.write(f"\r  {percent:5.1f}%  "
                                     f"{written / 1e6:6.1f}/{total / 1e6:.1f} MB")
                    sys.stderr.flush()
        if progress and total and sys.stderr.isatty():
            sys.stderr.write("\n")
    shutil.move(temporary, destination)
    return written, digest.hexdigest()


def pull_model(name: str = DEFAULT_MODEL, destination: Optional[str] = None,
               force: bool = False, progress: bool = True) -> str:
    """Download a model into the user data directory and return its path."""
    spec = _all_specs().get(name)
    if spec is None:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(_all_specs())}")

    target = destination or default_model_path(name)
    if os.path.exists(target) and not force:
        return os.path.abspath(target)
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)

    errors = []
    for url in spec.urls():
        try:
            if progress:
                print(f"downloading {spec.name} from {url}", file=sys.stderr)
            size, sha = _download(url, target, progress=progress)
            if size < 100_000:
                raise IOError(f"downloaded only {size} bytes; probably not a model")
            print(f"saved {target} ({size / 1e6:.1f} MB, sha256 {sha[:16]}…)",
                  file=sys.stderr)
            return os.path.abspath(target)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, IOError) as exc:
            errors.append(f"{url}: {exc}")
            continue
    raise RuntimeError("could not download the model:\n  " + "\n  ".join(errors))


def list_models(include_face: bool = True) -> List[Tuple[str, str, bool, str]]:
    """``(name, filename, installed, path)`` for each known model."""
    rows = []
    for name, spec in (_all_specs() if include_face else MODELS).items():
        found = find_model(name)
        rows.append((name, spec.filename, found is not None, found or ""))
    return rows
