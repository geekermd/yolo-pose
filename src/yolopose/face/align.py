"""Face alignment and embedding.

Alignment uses the standard ArcFace 5-point template and a similarity transform
estimated with the Umeyama method, then resamples through a bilinear grid warp.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scrfd import Face

__all__ = ["ARCFACE_TEMPLATE", "estimate_similarity", "warp_face",
           "align_face", "FaceEmbedder", "embedding_similarity"]

# Canonical 112x112 landmark positions used by ArcFace/InsightFace models.
ARCFACE_TEMPLATE: np.ndarray = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def estimate_similarity(source: np.ndarray, destination: np.ndarray,
                        estimate_scale: bool = True) -> np.ndarray:
    """Umeyama similarity transform mapping ``source`` points onto ``destination``.

    Returns a ``2x3`` matrix ``M`` such that ``destination ~= M @ [source, 1]``.
    """
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and destination must both be (N, 2)")

    count = source.shape[0]
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean

    covariance = destination_centered.T @ source_centered / count
    u, singular_values, vt = np.linalg.svd(covariance)

    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0

    rotation = u @ np.diag(sign) @ vt
    if estimate_scale:
        source_variance = (source_centered ** 2).sum() / count
        scale = (singular_values * sign).sum() / max(source_variance, 1e-12)
    else:
        scale = 1.0

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = destination_mean - scale * rotation @ source_mean
    return matrix.astype(np.float32)


def _grid_for(matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Inverse-sample grid (normalised, ``(1, H, W, 2)``) for an affine matrix."""
    height, width = int(size[0]), int(size[1])
    ys = (np.arange(height, dtype=np.float32) * 2.0 + 1.0) / height - 1.0
    xs = (np.arange(width, dtype=np.float32) * 2.0 + 1.0) / width - 1.0
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    # Destination pixel centre coordinates (unnormalised).
    dest_x = (grid_x + 1.0) * width / 2.0 - 0.5
    dest_y = (grid_y + 1.0) * height / 2.0 - 0.5
    flat = np.stack([dest_x.ravel(), dest_y.ravel(),
                     np.ones(dest_x.size, dtype=np.float32)], axis=0)
    inverse = np.linalg.inv(np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64))
    source_points = (inverse[:2] @ flat).astype(np.float32)
    return source_points.T.reshape(1, height, width, 2)


def warp_face(image: np.ndarray, matrix: np.ndarray,
              size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """Warp an RGB image with a ``2x3`` affine matrix into ``size``."""
    from ..core.runtime import _gather_spatial

    height, width = int(size[0]), int(size[1])
    grid = _grid_for(matrix, (height, width))
    # Normalise the source coordinates back into grid space for sampling.
    src = grid[0]
    gx = (src[:, :, 0] * 2.0 + 1.0) / image.shape[1] - 1.0
    gy = (src[:, :, 1] * 2.0 + 1.0) / image.shape[0] - 1.0

    tensor = image.astype(np.float32).transpose(2, 0, 1)[None]     # (1, 3, H, W)
    in_h, in_w = image.shape[:2]
    sample_x = ((gx + 1.0) * in_w - 1.0) / 2.0
    sample_y = ((gy + 1.0) * in_h - 1.0) / 2.0

    x0 = np.clip(np.floor(sample_x).astype(np.int64), 0, in_w - 1)
    y0 = np.clip(np.floor(sample_y).astype(np.int64), 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    wx = (sample_x - np.floor(sample_x)).astype(np.float32)[None, None]
    wy = (sample_y - np.floor(sample_y)).astype(np.float32)[None, None]

    top_left = _gather_spatial(tensor, y0[None], x0[None])
    top_right = _gather_spatial(tensor, y0[None], x1[None])
    bottom_left = _gather_spatial(tensor, y1[None], x0[None])
    bottom_right = _gather_spatial(tensor, y1[None], x1[None])
    top = top_left * (1 - wx) + top_right * wx
    bottom = bottom_left * (1 - wx) + bottom_right * wx
    warped = top * (1 - wy) + bottom * wy
    return np.clip(warped[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)


def align_face(image: np.ndarray, face: Face,
               size: Tuple[int, int] = (112, 112),
               template: np.ndarray = ARCFACE_TEMPLATE) -> np.ndarray:
    """Align a detected face to the canonical pose."""
    destination = template if size == (112, 112) else \
        template * (np.asarray(size, dtype=np.float32) / 112.0)
    matrix = estimate_similarity(face.landmarks, destination)
    return warp_face(image, matrix, size)


class FaceEmbedder:
    """Turns aligned 112x112 faces into L2-normalised embedding vectors."""

    def __init__(self, model_path: str, backend: str = "auto",
                 input_size: Tuple[int, int] = (112, 112), verbose: bool = False,
                 provider: str = "auto"):
        from ..vision.detector import _OnnxRuntimeEngine
        from ..core.runtime import NumpyEngine

        self.model_path = model_path
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.verbose = verbose

        wanted = backend.lower()
        self.backend_name = "numpy"
        engine = None
        if wanted in ("auto", "onnxruntime", "ort"):
            try:
                engine = _OnnxRuntimeEngine(model_path, provider=provider)
                self.backend_name = "onnxruntime"
            except Exception as exc:
                if wanted not in ("auto",):
                    raise
                if verbose:
                    print(f"[yolo-pose] onnxruntime unavailable for embeddings ({exc})")
        if engine is None:
            engine = NumpyEngine(model_path, verbose=verbose)
        self.engine = engine

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        """Return the embedding for one aligned face crop."""
        return self.embed_batch(aligned[None])[0]

    def embed_batch(self, aligned: np.ndarray) -> np.ndarray:
        """Embed a batch of aligned crops given as ``(N, H, W, 3)``."""
        batch = np.asarray(aligned)
        if batch.ndim == 3:
            batch = batch[None]
        tensor = batch.astype(np.float32)
        if tensor.max() > 1.5:
            tensor = (tensor - 127.5) / 127.5
        tensor = np.ascontiguousarray(tensor.transpose(0, 3, 1, 2))
        outputs = self.engine.run({self.engine.input_name: tensor})
        embeddings = np.asarray(next(iter(outputs.values())), dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def describe(self) -> str:
        return (f"face embedder : {self.model_path}\n"
                f"backend       : {self.backend_name}\n"
                f"input size    : {self.input_size[0]}x{self.input_size[1]}")


def embedding_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Cosine similarity between (batches of) L2-normalised embeddings."""
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    if b.ndim == 1:
        b = b[None]
    return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)) @ \
           (b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)).T
