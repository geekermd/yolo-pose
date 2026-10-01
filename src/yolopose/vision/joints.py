"""Joint-angle analysis built on COCO-17 keypoints.

Angles are computed as the angle at the *middle* vertex of three keypoints, in
degrees.  A joint is only reported when all three contributing keypoints clear
the confidence threshold, so downstream users never see angles derived from
hallucinated keypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .pose import KEYPOINT_INDEX, KEYPOINT_NAMES, Pose


@dataclass(frozen=True)
class JointDefinition:
    """A named angle: the vertex plus the two rays that define it."""

    name: str
    vertex: str
    first: str
    second: str
    label: str


# Left/right joints are declared centre-line-free so a missing side simply drops out.
JOINT_DEFINITIONS: Tuple[JointDefinition, ...] = (
    # Convention (standard goniometry): the measured joint is the VERTEX, and the
    # two rays point at the body parts either side of it.
    #   knee    = thigh vs shin      (180 deg when the leg is straight)
    #   elbow   = upper arm vs forearm
    #   hip     = torso vs thigh
    #   shoulder= torso vs upper arm
    #   ankle   = shin vs (virtual) toe
    JointDefinition("left_knee", "left_knee", "left_hip", "left_ankle", "左膝"),
    JointDefinition("right_knee", "right_knee", "right_hip", "right_ankle", "右膝"),
    JointDefinition("left_elbow", "left_elbow", "left_shoulder", "left_wrist",
                    "左肘"),
    JointDefinition("right_elbow", "right_elbow", "right_shoulder", "right_wrist",
                    "右肘"),
    JointDefinition("left_hip", "left_hip", "left_shoulder", "left_knee", "左髋"),
    JointDefinition("right_hip", "right_hip", "right_shoulder", "right_knee",
                    "右髋"),
    JointDefinition("left_shoulder", "left_shoulder", "left_elbow", "right_shoulder",
                    "左肩"),
    JointDefinition("right_shoulder", "right_shoulder", "right_elbow",
                    "left_shoulder", "右肩"),
    JointDefinition("left_ankle", "left_ankle", "left_knee", "left_big_toe", "左踝"),
    JointDefinition("right_ankle", "right_ankle", "right_knee", "right_big_toe",
                    "右踝"),
    # Neck/head orientation: head-top axis vs the shoulder->hip body axis.
    JointDefinition("neck", "mid_shoulder", "nose", "mid_hip", "颈"),
    # Torso alignment: shoulder line vs the line down to the feet.
    JointDefinition("torso", "mid_hip", "mid_shoulder", "mid_ankle", "躯干"),
)

# Virtual points keep the neck/torso angles meaningful on a COCO-17 skeleton,
# which has no explicit neck or pelvis keypoint.
_VIRTUAL_POINTS = {
    # Midpoints of keypoint pairs: COCO-17 has no neck or pelvis keypoint.
    "mid_shoulder": ("left_shoulder", "right_shoulder"),
    "mid_hip": ("left_hip", "right_hip"),
    "mid_ankle": ("left_ankle", "right_ankle"),
}

# Toe proxies are derived from the ankle/knee instead of a keypoint pair.
_TOE_PROXIES = {"left_big_toe", "right_big_toe"}

# Distance used for the synthetic toe / head reference points.
_VIRTUAL_TOE_OFFSET = 40.0


class JointError(RuntimeError):
    """Raised when an angle cannot be computed."""


def angle_at(vertex: Sequence[float], first: Sequence[float],
             second: Sequence[float]) -> float:
    """Angle in degrees at ``vertex`` between the rays to ``first``/``second``."""
    v = np.asarray([vertex[0], vertex[1]], dtype=np.float64)
    a = np.asarray([first[0], first[1]], dtype=np.float64) - v
    b = np.asarray([second[0], second[1]], dtype=np.float64) - v
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        raise JointError("degenerate limb: zero-length segment")
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


@dataclass
class JointAngles:
    """Computed angles for one person."""

    angles: Dict[str, float]
    skipped: Dict[str, str]

    def __getitem__(self, name: str) -> float:
        return self.angles[name]

    def get(self, name: str, default: Optional[float] = None) -> Optional[float]:
        return self.angles.get(name, default)

    def as_dict(self) -> Dict[str, object]:
        return {
            "angles": {k: round(v, 2) for k, v in self.angles.items()},
            "skipped": dict(self.skipped),
        }

    def summary(self, threshold: float = 0.5) -> str:
        if not self.angles:
            return "no joint could be measured"
        parts = [f"{k}={v:.1f}°" for k, v in sorted(self.angles.items())]
        return ", ".join(parts)


def _virtual_point(pose: Pose, name: str, offset: float) -> Tuple[float, float]:
    """Synthesise a point that COCO-17 does not provide.

    * ``left/right_big_toe`` — a toe extended from the ankle along the shin, so
      the ankle angle reflects plantar/dorsiflexion.
    * ``*_midpoint`` / ``*_base`` — the midpoint of a keypoint pair (neck, hips).
    """
    if name in _TOE_PROXIES:
        side = "left" if name.startswith("left") else "right"
        ankle = KEYPOINT_INDEX[f"{side}_ankle"]
        knee = KEYPOINT_INDEX[f"{side}_knee"]
        ax, ay = pose.keypoints[ankle]
        kx, ky = pose.keypoints[knee]
        direction = np.asarray([ax - kx, ay - ky], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return float(ax), float(ay + offset)
        direction /= norm
        return float(ax + direction[0] * offset), float(ay + direction[1] * offset)

    pair = _VIRTUAL_POINTS.get(name)
    if pair is None:
        raise JointError(f"no virtual point named {name!r}")
    first = pose.keypoints[KEYPOINT_INDEX[pair[0]]]
    second = pose.keypoints[KEYPOINT_INDEX[pair[1]]]
    midpoint = (np.asarray(first, dtype=np.float64)
                + np.asarray(second, dtype=np.float64)) / 2.0
    return float(midpoint[0]), float(midpoint[1])


def compute_joint_angles(pose: Pose, keypoint_threshold: float = 0.5,
                         definitions: Iterable[JointDefinition] = JOINT_DEFINITIONS
                         ) -> JointAngles:
    """Compute every joint angle that the visible keypoints support."""
    scores = pose.keypoint_scores
    points = pose.keypoints

    def required_keypoints(name: str) -> Tuple[str, ...]:
        if name in _TOE_PROXIES:
            side = "left" if name.startswith("left") else "right"
            return (f"{side}_ankle", f"{side}_knee")
        if name in _VIRTUAL_POINTS:
            return _VIRTUAL_POINTS[name]
        return (name,)

    def usable(name: str) -> bool:
        needed = required_keypoints(name)
        return all(
            KEYPOINT_INDEX.get(k) is not None
            and scores[KEYPOINT_INDEX[k]] >= keypoint_threshold
            for k in needed)

    def point_of(name: str) -> Tuple[float, float]:
        if name in _TOE_PROXIES or name in _VIRTUAL_POINTS:
            return _virtual_point(pose, name, _VIRTUAL_TOE_OFFSET)
        index = KEYPOINT_INDEX[name]
        return float(points[index, 0]), float(points[index, 1])

    angles: Dict[str, float] = {}
    skipped: Dict[str, str] = {}
    for definition in definitions:
        missing = [n for n in (definition.vertex, definition.first, definition.second)
                   if not usable(n)]
        if missing:
            skipped[definition.name] = "low confidence: " + ", ".join(missing)
            continue
        try:
            angles[definition.name] = angle_at(point_of(definition.vertex),
                                               point_of(definition.first),
                                               point_of(definition.second))
        except JointError as exc:
            skipped[definition.name] = str(exc)
    return JointAngles(angles=angles, skipped=skipped)


def torsoSkeleton(pose: Pose, threshold: float = 0.5) -> Optional[np.ndarray]:
    """Centre line of the torso (mid-shoulders -> mid-hips), or ``None``."""
    scores = pose.keypoint_scores
    needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(scores[KEYPOINT_INDEX[n]] < threshold for n in needed):
        return None
    shoulders = (pose.keypoints[KEYPOINT_INDEX["left_shoulder"]]
                 + pose.keypoints[KEYPOINT_INDEX["right_shoulder"]]) / 2.0
    hips = (pose.keypoints[KEYPOINT_INDEX["left_hip"]]
            + pose.keypoints[KEYPOINT_INDEX["right_hip"]]) / 2.0
    return np.stack([shoulders, hips])


def angle_table(pose: Pose, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """``[(name, chinese label, degrees)]`` sorted by name, for reporting."""
    result = compute_joint_angles(pose, threshold)
    rows: List[Tuple[str, str, float]] = []
    for definition in JOINT_DEFINITIONS:
        if definition.name in result.angles:
            rows.append((definition.name, definition.label,
                         result.angles[definition.name]))
    return rows
