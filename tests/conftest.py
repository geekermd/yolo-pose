"""Shared pytest fixtures: a tiny synthetic ONNX runtime and sample images."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
POSE_MODEL = os.path.join(MODEL_DIR, "yolov8n-pose.onnx")
FACE_DETECTOR = os.path.join(MODEL_DIR, "scrfd_500m.onnx")
FACE_EMBEDDER = os.path.join(MODEL_DIR, "w600k_mbf.onnx")


def _has(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 100_000


requires_pose_model = pytest.mark.skipif(
    not _has(POSE_MODEL), reason="pose model not installed")
requires_face_models = pytest.mark.skipif(
    not (_has(FACE_DETECTOR) and _has(FACE_EMBEDDER)),
    reason="face models not installed")


@pytest.fixture(scope="session")
def pose_model() -> str:
    if not _has(POSE_MODEL):
        pytest.skip("pose model not installed")
    return POSE_MODEL


@pytest.fixture(scope="session")
def face_models():
    if not (_has(FACE_DETECTOR) and _has(FACE_EMBEDDER)):
        pytest.skip("face models not installed")
    return FACE_DETECTOR, FACE_EMBEDDER


@pytest.fixture
def gradient_image() -> np.ndarray:
    """A colourful 240x320 RGB test image."""
    height, width = 240, 320
    ys, xs = np.mgrid[0:height, 0:width]
    image = np.stack([
        (xs * 255 // width),
        (ys * 255 // height),
        ((xs + ys) * 255 // (width + height)),
    ], axis=2).astype(np.uint8)
    image[40:120, 60:200] = (220, 30, 30)
    return image


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """A crude but detectable face-like image for pipeline smoke tests."""
    height, width = 240, 240
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    # Head
    ys, xs = np.mgrid[0:height, 0:width]
    head = ((ys - 130) ** 2 / 90.0 ** 2 + (xs - 120) ** 2 / 70.0 ** 2) <= 1.0
    image[head] = (235, 200, 175)
    # Eyes
    for centre in ((98, 110), (142, 110)):
        eye = ((ys - centre[1]) ** 2 + (xs - centre[0]) ** 2) <= 8 ** 2
        image[eye] = (40, 40, 40)
    # Mouth
    mouth = ((ys - 175) ** 2 / 8.0 ** 2 + (xs - 120) ** 2 / 30.0 ** 2) <= 1.0
    image[mouth] = (150, 60, 60)
    return image
