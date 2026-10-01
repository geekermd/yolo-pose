"""YOLOv8-pose decoding: raw network output -> human poses.

The network emits a ``(1, 56, N)`` tensor whose channels are
``[cx, cy, w, h, score, kx0, ky0, kc0, ... kx16, ky16, kc16]`` in *letterboxed
pixel* coordinates, where the box centre/size are already in input-image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# COCO-17 keypoint names, in the order emitted by the model.
KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Skeleton edges as index pairs; used by the renderer and the angle module.
SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
)

# Edges that make up the left / right side, for colour coding.
LEFT_EDGES = frozenset({(0, 1), (1, 3), (5, 7), (7, 9), (5, 11), (11, 13), (13, 15)})
RIGHT_EDGES = frozenset({(0, 2), (2, 4), (6, 8), (8, 10), (6, 12), (12, 14), (14, 16)})


@dataclass
class Pose:
    """A single detected person."""

    box: np.ndarray            # (4,) float32 xyxy in source-image pixels
    score: float
    keypoints: np.ndarray      # (17, 2) float32 xy in source-image pixels
    keypoint_scores: np.ndarray  # (17,) float32 in [0, 1]

    def visible(self, threshold: float = 0.5) -> np.ndarray:
        return self.keypoint_scores >= threshold

    def keypoint(self, name: str) -> Tuple[float, float, float]:
        index = KEYPOINT_INDEX[name]
        x, y = self.keypoints[index]
        return float(x), float(y), float(self.keypoint_scores[index])

    def as_dict(self) -> dict:
        return {
            "box": [float(v) for v in self.box],
            "score": float(self.score),
            "keypoints": {
                name: {
                    "x": float(self.keypoints[i, 0]),
                    "y": float(self.keypoints[i, 1]),
                    "score": float(self.keypoint_scores[i]),
                }
                for i, name in enumerate(KEYPOINT_NAMES)
            },
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h]`` to ``[x1, y1, x2, y2]``."""
    out = np.empty_like(boxes, dtype=np.float32)
    half_w = boxes[..., 2] / 2.0
    half_h = boxes[..., 3] / 2.0
    out[..., 0] = boxes[..., 0] - half_w
    out[..., 1] = boxes[..., 1] - half_h
    out[..., 2] = boxes[..., 0] + half_w
    out[..., 3] = boxes[..., 1] + half_h
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of ``xyxy`` boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45,
        max_detections: int = 300) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices."""
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
        ious = box_iou(boxes[best:best + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #

def decode_poses(output: np.ndarray, info, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, max_detections: int = 100,
                 max_keypoints: int = 17) -> List[Pose]:
    """Turn one raw model output into a list of :class:`Pose` objects.

    ``info`` is a :class:`~yolopose.vision.imaging.LetterboxInfo` used to map
    letterbox coordinates back onto the original image.
    """
    from .imaging import map_box

    if output.ndim == 3:
        prediction = output[0]           # (C, N)
    elif output.ndim == 2:
        prediction = output
    else:
        raise ValueError(f"unexpected model output rank {output.ndim}")
    # Expect (C, N) with C = 5 + 3*K.  Fall back to transposing (N, C) layouts
    # only when the current orientation cannot hold a keypoint channel set.
    channels_first = prediction.shape[0]
    plausible = channels_first >= 5 and (channels_first - 5) % 3 == 0
    if not plausible and prediction.shape[1] >= 5:
        prediction = prediction.T

    channels, anchors = prediction.shape
    if channels < 5:
        raise ValueError(f"model output has only {channels} channels")

    expected = 5 + 3 * max_keypoints
    if channels != expected:
        # Trust the metadata rather than the caller when they disagree.
        max_keypoints = max(1, (channels - 5) // 3)

    scores = prediction[4].astype(np.float32)
    candidates = np.flatnonzero(scores >= conf_threshold)
    if candidates.size == 0:
        return []

    boxes_net = xywh_to_xyxy(prediction[:4, candidates].T.astype(np.float32))
    kp_raw = (prediction[5:5 + 3 * max_keypoints, candidates].T
              .reshape(-1, max_keypoints, 3).astype(np.float32))
    kp_xy_net = kp_raw[:, :, :2]
    kp_conf = kp_raw[:, :, 2]

    # Scores come out of the graph post-sigmoid already, but clamp defensively.
    kp_conf = np.clip(kp_conf, 0.0, 1.0)
    scores = np.clip(scores[candidates], 0.0, 1.0)

    keep = nms(boxes_net, scores, iou_threshold, max_detections)
    boxes_net = boxes_net[keep]
    kp_xy_net = kp_xy_net[keep]
    kp_conf = kp_conf[keep]
    scores = scores[keep]

    boxes_src = map_box(boxes_net, info)
    scale = info.scale
    pad_x, pad_y = info.pad_x, info.pad_y
    kp_x = (kp_xy_net[:, :, 0] - pad_x) / scale
    kp_y = (kp_xy_net[:, :, 1] - pad_y) / scale
    kp_x = np.clip(kp_x, 0, info.src_width - 1)
    kp_y = np.clip(kp_y, 0, info.src_height - 1)
    keypoints = np.stack([kp_x, kp_y], axis=2).astype(np.float32)

    poses: List[Pose] = []
    for i in range(len(scores)):
        poses.append(Pose(box=boxes_src[i].astype(np.float32),
                          score=float(scores[i]),
                          keypoints=keypoints[i],
                          keypoint_scores=kp_conf[i].astype(np.float32)))
    return poses


def count_visible(pose: Pose, threshold: float = 0.5) -> int:
    return int(np.count_nonzero(pose.keypoint_scores >= threshold))
