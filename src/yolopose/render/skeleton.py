"""Skeleton and overlay rendering with NumPy only.

Lines and circles are rasterised through a signed-distance field and composited
with alpha coverage, which gives clean anti-aliased edges without OpenCV or
Pillow.  Text uses a compact built-in 5x7 bitmap font so labels always render.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..vision.pose import (KEYPOINT_NAMES, LEFT_EDGES, RIGHT_EDGES,
                           SKELETON_EDGES, Pose)

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LEFT_COLOUR = (255, 128, 0)      # orange
RIGHT_COLOUR = (0, 200, 255)     # cyan
CENTER_COLOUR = (0, 255, 0)      # green
BOX_COLOUR = (60, 220, 60)
KEYPOINT_FILL = (255, 255, 255)

# 5x7 font: each glyph is 7 rows, each row a 5-bit mask (MSB = leftmost pixel).
FONT: Dict[str, Tuple[int, ...]] = {
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    ",": (0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    "_": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "%": (0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13),
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
    "?": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04),
    "!": (0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04),
    "'": (0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    "=": (0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00),
}


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

def _blend(canvas: np.ndarray, colour: Sequence[int], alpha: np.ndarray,
           y0: int, x0: int) -> None:
    """Alpha-composite ``colour`` onto a canvas patch."""
    if alpha.size == 0:
        return
    height, width = canvas.shape[:2]
    patch_h, patch_w = alpha.shape
    y1 = min(height, y0 + patch_h)
    x1 = min(width, x0 + patch_w)
    y0c = max(0, y0)
    x0c = max(0, x0)
    if y1 <= y0c or x1 <= x0c:
        return
    sub = alpha[y0c - y0:y1 - y0, x0c - x0:x1 - x0]
    if sub.size == 0:
        return
    target = canvas[y0c:y1, x0c:x1].astype(np.float32)
    colour_array = np.asarray(colour, dtype=np.float32).reshape(1, 1, 3)
    blended = target * (1.0 - sub[:, :, None]) + colour_array * sub[:, :, None]
    canvas[y0c:y1, x0c:x1] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_line(canvas: np.ndarray, start: Sequence[float], end: Sequence[float],
              colour: Sequence[int], width: float = 2.0,
              alpha: float = 1.0) -> None:
    """Anti-aliased thick line (signed-distance based)."""
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    half = max(width, 1.0) / 2.0
    margin = int(np.ceil(half + 1.5))

    left = int(np.floor(min(x0, x1))) - margin
    right = int(np.ceil(max(x0, x1))) + margin
    top = int(np.floor(min(y0, y1))) - margin
    bottom = int(np.ceil(max(y0, y1))) + margin

    height, width_px = canvas.shape[:2]
    left_c, right_c = max(0, left), min(width_px, right)
    top_c, bottom_c = max(0, top), min(height, bottom)
    if right_c <= left_c or bottom_c <= top_c:
        return

    xs = np.arange(left_c, right_c, dtype=np.float32) + 0.5
    ys = np.arange(top_c, bottom_c, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        distance = np.hypot(grid_x - x0, grid_y - y0)
    else:
        t = np.clip(((grid_x - x0) * dx + (grid_y - y0) * dy) / length_sq, 0.0, 1.0)
        distance = np.hypot(grid_x - (x0 + t * dx), grid_y - (y0 + t * dy))

    coverage = np.clip(half + 0.5 - distance, 0.0, 1.0) * float(alpha)
    _blend(canvas, colour, coverage, top_c, left_c)


def draw_circle(canvas: np.ndarray, centre: Sequence[float], radius: float,
                colour: Sequence[int], fill: bool = True, alpha: float = 1.0,
                width: float = 2.0) -> None:
    """Anti-aliased filled or outlined circle."""
    cx, cy = float(centre[0]), float(centre[1])
    r = max(float(radius), 0.5)
    margin = int(np.ceil(r + width + 2))
    left = int(np.floor(cx)) - margin
    right = int(np.ceil(cx)) + margin
    top = int(np.floor(cy)) - margin
    bottom = int(np.ceil(cy)) + margin

    height, width_px = canvas.shape[:2]
    left_c, right_c = max(0, left), min(width_px, right)
    top_c, bottom_c = max(0, top), min(height, bottom)
    if right_c <= left_c or bottom_c <= top_c:
        return

    xs = np.arange(left_c, right_c, dtype=np.float32) + 0.5
    ys = np.arange(top_c, bottom_c, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance = np.hypot(grid_x - cx, grid_y - cy)

    if fill:
        coverage = np.clip(r + 0.5 - distance, 0.0, 1.0)
    else:
        band = np.abs(distance - r)
        coverage = np.clip(width / 2.0 + 0.5 - band, 0.0, 1.0)
    _blend(canvas, colour, coverage * float(alpha), top_c, left_c)


def draw_rectangle(canvas: np.ndarray, box: Sequence[float], colour: Sequence[int],
                   width: float = 2.0, alpha: float = 1.0) -> None:
    """Axis-aligned rectangle outline."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    draw_line(canvas, (x1, y1), (x2, y1), colour, width, alpha)
    draw_line(canvas, (x2, y1), (x2, y2), colour, width, alpha)
    draw_line(canvas, (x2, y2), (x1, y2), colour, width, alpha)
    draw_line(canvas, (x1, y2), (x1, y1), colour, width, alpha)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #

def text_size(text: str, scale: int = 2) -> Tuple[int, int]:
    width = 0 if not text else len(text) * 6 * scale - scale
    return width, 7 * scale


def draw_text(canvas: np.ndarray, text: str, origin: Sequence[float],
              colour: Sequence[int] = WHITE, scale: int = 2,
              alpha: float = 1.0) -> None:
    """Render ASCII text with the built-in 5x7 bitmap font.

    Characters outside the font are replaced by ``?``; CJK labels are drawn via
    a separate routine by the caller (see :func:`draw_label`).
    """
    scale = max(1, int(scale))
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    height, width = canvas.shape[:2]

    for character in text:
        glyph = FONT.get(character.upper())
        if glyph is None:
            glyph = FONT.get(character)
        if glyph is None:
            glyph = FONT["?"]
        for row, mask in enumerate(glyph):
            for col in range(5):
                if not (mask >> (4 - col)) & 1:
                    continue
                px = x + col * scale
                py = y + row * scale
                if px < 0 or py < 0 or px + scale > width or py + scale > height:
                    continue
                patch = canvas[py:py + scale, px:px + scale].astype(np.float32)
                target = np.asarray(colour, dtype=np.float32).reshape(1, 1, 3)
                blended = patch * (1.0 - alpha) + target * alpha
                canvas[py:py + scale, px:px + scale] = np.clip(blended, 0, 255).astype(np.uint8)
        x += 6 * scale


def draw_label(canvas: np.ndarray, text: str, origin: Sequence[float],
               colour: Sequence[int] = WHITE, scale: int = 2,
               background: Optional[Sequence[int]] = (0, 0, 0),
               alpha: float = 0.6) -> None:
    """Draw text with an optional translucent background plate."""
    x, y = int(round(origin[0])), int(round(origin[1]))
    width, height = text_size(text, scale)
    if background is not None:
        canvas_h, canvas_w = canvas.shape[:2]
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(canvas_w, x + width + 2), min(canvas_h, y + height + 2)
        if x1 > x0 and y1 > y0:
            patch = canvas[y0:y1, x0:x1].astype(np.float32)
            target = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
            canvas[y0:y1, x0:x1] = np.clip(
                patch * (1 - alpha) + target * alpha, 0, 255).astype(np.uint8)
    draw_text(canvas, text, (x, y), colour, scale)


# --------------------------------------------------------------------------- #
# high level overlays
# --------------------------------------------------------------------------- #

def draw_skeleton(canvas: np.ndarray, pose: Pose, keypoint_threshold: float = 0.4,
                  line_width: float = 2.0, radius: float = 3.0) -> None:
    """Draw one person's skeleton and keypoints."""
    scores = pose.keypoint_scores
    points = pose.keypoints

    for a, b in SKELETON_EDGES:
        if scores[a] < keypoint_threshold or scores[b] < keypoint_threshold:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in LEFT_EDGES:
            colour = LEFT_COLOUR
        elif edge in RIGHT_EDGES:
            colour = RIGHT_COLOUR
        else:
            colour = CENTER_COLOUR
        draw_line(canvas, points[a], points[b], colour, line_width,
                  alpha=0.9)

    for index in range(len(KEYPOINT_NAMES)):
        if scores[index] < keypoint_threshold:
            continue
        draw_circle(canvas, points[index], radius, KEYPOINT_FILL, fill=True, alpha=0.95)
        draw_circle(canvas, points[index], radius, BLACK, fill=False, width=1.0, alpha=0.5)


def draw_pose(canvas: np.ndarray, pose: Pose, keypoint_threshold: float = 0.4,
              show_box: bool = True, show_score: bool = True,
              label: Optional[str] = None) -> None:
    """Box + skeleton + score for a single person."""
    if show_box:
        draw_rectangle(canvas, pose.box, BOX_COLOUR, width=2.0, alpha=0.9)
    draw_skeleton(canvas, pose, keypoint_threshold)

    if show_score or label:
        text = label if label else f"person {pose.score * 100:.0f}%"
        origin = (pose.box[0], max(12, pose.box[1] - 16))
        draw_label(canvas, text, origin, WHITE, scale=2)


def draw_angle_labels(canvas: np.ndarray, pose: Pose, angles,
                      keypoint_threshold: float = 0.5,
                      scale: int = 1) -> None:
    """Annotate measured joints with their angle in degrees.

    Labels are placed slightly offset from the joint so they do not cover the
    skeleton itself.
    """
    from ..vision.pose import KEYPOINT_INDEX

    vertices = {
        "left_elbow": "left_elbow", "right_elbow": "right_elbow",
        "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
        "left_hip": "left_hip", "right_hip": "right_hip",
        "left_knee": "left_knee", "right_knee": "right_knee",
        "left_ankle": "left_ankle", "right_ankle": "right_ankle",
    }
    for name, degrees in angles.angles.items():
        vertex_name = vertices.get(name)
        if vertex_name is None:
            continue
        index = KEYPOINT_INDEX[vertex_name]
        if pose.keypoint_scores[index] < keypoint_threshold:
            continue
        x, y = pose.keypoints[index]
        text = f"{degrees:.0f}"
        draw_label(canvas, text, (x + 8, y - 4), WHITE, scale=scale,
                   background=(20, 20, 20), alpha=0.65)


def draw_hud(canvas: np.ndarray, lines: Sequence[str], origin=(10, 10),
             scale: int = 2) -> None:
    """Top-left status block, e.g. FPS and detection count."""
    x, y = origin
    for i, line in enumerate(lines):
        draw_label(canvas, line, (x, y + i * (7 * scale + 6)), WHITE, scale=scale,
                   background=(0, 0, 0), alpha=0.55)
