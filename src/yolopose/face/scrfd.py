"""Face detection with SCRFD (Sample and Computation Redistribution for Face Detection).

The ONNX graph emits, for each of three feature strides (8/16/32):
``score``, ``bbox`` (distance regression, 4 channels) and ``kps`` (5 landmarks,
10 channels).  Output tensor names are opaque numbers in the official exports,
so heads are identified by their shape rather than by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Face", "FaceDetector", "decode_scrfd"]

# Anchor counts per location for SCRFD (2 anchors per position).
_NUM_ANCHORS = 2


@dataclass
class Face:
    """A detected face."""

    box: np.ndarray            # (4,) float32 xyxy in source-image pixels
    score: float
    landmarks: np.ndarray      # (5, 2) float32: eyes, nose, mouth corners

    def as_dict(self) -> dict:
        return {
            "box": [round(float(v), 2) for v in self.box],
            "score": round(float(self.score), 4),
            "landmarks": [[round(float(x), 2), round(float(y), 2)]
                          for x, y in self.landmarks],
        }


def _distance2bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Convert (l, t, r, b) distances around anchor points into xyxy boxes."""
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def _distance2kps(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Convert landmark distances around anchors into absolute coordinates."""
    outputs = []
    for index in range(0, distances.shape[1], 2):
        outputs.append(points[:, 0] + distances[:, index])
        outputs.append(points[:, 1] + distances[:, index + 1])
    return np.stack(outputs, axis=1)


def _anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    """Anchor centres for one feature level, repeated for each anchor."""
    ys, xs = np.mgrid[:height, :width]
    centers = np.stack([xs * stride, ys * stride], axis=-1).astype(np.float32)
    centers = centers.reshape(-1, 2)
    return np.repeat(centers, _NUM_ANCHORS, axis=0)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4,
        max_detections: int = 1000) -> np.ndarray:
    """Greedy IoU suppression; returns kept indices."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_detections:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_best = ((boxes[best, 2] - boxes[best, 0])
                     * (boxes[best, 3] - boxes[best, 1]))
        area_rest = ((boxes[rest, 2] - boxes[rest, 0])
                     * (boxes[rest, 3] - boxes[rest, 1]))
        union = area_best + area_rest - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def _classify_outputs(outputs: Dict[str, np.ndarray],
                      input_size: Tuple[int, int]
                      ) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Group raw output tensors into ``(scores, bbox, kps)`` per stride.

    SCRFD exports name tensors with numeric ids, so heads are matched by the
    number of columns: ``1`` (score), ``4`` (box distances) and ``10`` (5
    landmarks).  Levels are paired by their row count (one row per anchor).
    """
    scored: Dict[int, np.ndarray] = {}
    boxed: Dict[int, np.ndarray] = {}
    keypointed: Dict[int, np.ndarray] = {}

    for tensor in outputs.values():
        array = np.asarray(tensor)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            continue
        rows, columns = array.shape
        if columns == 1:
            scored[rows] = array[:, 0]
        elif columns == 4:
            boxed[rows] = array
        elif columns == 10:
            keypointed[rows] = array

    levels = []
    for rows in sorted(scored, reverse=True):
        if rows in boxed:
            levels.append((scored[rows], boxed[rows], keypointed.get(rows)))
    return levels


def decode_scrfd(outputs: Dict[str, np.ndarray], input_size: Tuple[int, int],
                 score_threshold: float = 0.5, nms_threshold: float = 0.4,
                 max_detections: int = 1000
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode SCRFD outputs into ``(boxes, scores, landmarks)``.

    Coordinates are in network-input pixels; the caller scales them back to the
    original image.
    """
    height, width = int(input_size[1]), int(input_size[0])
    levels = _classify_outputs(outputs, (width, height))
    if not levels:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0, 5, 2), np.float32))

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_kps: List[np.ndarray] = []

    # Infer the stride of each level from its anchor count.
    for scores, boxes, kps in levels:
        rows = scores.shape[0]
        per_level = rows // _NUM_ANCHORS
        # Expected counts for strides 8, 16, 32 on the given input size.
        stride = None
        for candidate in (8, 16, 32):
            feature_h = (height + candidate - 1) // candidate
            feature_w = (width + candidate - 1) // candidate
            if feature_h * feature_w == per_level:
                stride = candidate
                break
        if stride is None:
            continue

        feature_h = (height + stride - 1) // stride
        feature_w = (width + stride - 1) // stride
        centers = _anchor_centers(feature_h, feature_w, stride)
        if centers.shape[0] != rows:
            continue

        keep = np.flatnonzero(scores >= score_threshold)
        if keep.size == 0:
            continue
        decoded = _distance2bbox(centers[keep], boxes[keep] * stride)
        all_boxes.append(decoded)
        all_scores.append(scores[keep])
        if kps is not None:
            all_kps.append(_distance2kps(centers[keep], kps[keep] * stride).reshape(-1, 5, 2))

    if not all_boxes:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0, 5, 2), np.float32))

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    landmarks = (np.concatenate(all_kps, axis=0) if all_kps
                 else np.zeros((boxes.shape[0], 5, 2), np.float32))

    order = scores.argsort()[::-1][:max_detections * 4]
    boxes, scores, landmarks = boxes[order], scores[order], landmarks[order]
    keep = nms(boxes, scores, nms_threshold, max_detections)
    return boxes[keep], scores[keep], landmarks[keep]


class FaceDetector:
    """SCRFD face detector running on the shared ONNX runtime."""

    def __init__(self, model_path: str, backend: str = "auto",
                 input_size: Tuple[int, int] = (640, 640),
                 score_threshold: float = 0.5, nms_threshold: float = 0.4,
                 max_detections: int = 100, verbose: bool = False,
                 provider: str = "auto"):
        from ..vision.detector import _OnnxRuntimeEngine
        from ..core.runtime import NumpyEngine

        self.model_path = model_path
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_detections = int(max_detections)
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
                    print(f"[yolo-pose] onnxruntime unavailable for faces ({exc})")
        if engine is None:
            engine = NumpyEngine(model_path, verbose=verbose)
        self.engine = engine

    # -- inference ---------------------------------------------------------- #
    def detect(self, image: np.ndarray, score_threshold: Optional[float] = None
               ) -> List[Face]:
        """Detect faces in an RGB ``uint8`` image."""
        from ..vision.imaging import letterbox

        threshold = (self.score_threshold if score_threshold is None
                     else float(score_threshold))
        height, width = image.shape[:2]
        padded, info = letterbox(image, self.input_size)
        tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray((tensor - 0.5) / 0.5)

        outputs = self.engine.run({self.engine.input_name: tensor})
        boxes, scores, landmarks = decode_scrfd(
            outputs, self.input_size, score_threshold=threshold,
            nms_threshold=self.nms_threshold, max_detections=self.max_detections)

        if boxes.shape[0] == 0:
            return []

        scaled = boxes.copy()
        scaled[:, [0, 2]] = (scaled[:, [0, 2]] - info.pad_x) / info.scale
        scaled[:, [1, 3]] = (scaled[:, [1, 3]] - info.pad_y) / info.scale
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, width - 1)
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, height - 1)

        landmark_scaled = landmarks.copy()
        landmark_scaled[:, :, 0] = (landmark_scaled[:, :, 0] - info.pad_x) / info.scale
        landmark_scaled[:, :, 1] = (landmark_scaled[:, :, 1] - info.pad_y) / info.scale
        landmark_scaled[:, :, 0] = np.clip(landmark_scaled[:, :, 0], 0, width - 1)
        landmark_scaled[:, :, 1] = np.clip(landmark_scaled[:, :, 1], 0, height - 1)

        return [Face(box=scaled[i].astype(np.float32), score=float(scores[i]),
                     landmarks=landmark_scaled[i].astype(np.float32))
                for i in range(scaled.shape[0])]

    def describe(self) -> str:
        return (f"face detector : SCRFD ({self.model_path})\n"
                f"backend       : {self.backend_name}\n"
                f"input size    : {self.input_size[0]}x{self.input_size[1]}\n"
                f"score / nms   : {self.score_threshold} / {self.nms_threshold}")
