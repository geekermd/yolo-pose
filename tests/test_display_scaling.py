"""Display-scaling tests: the enlarged window must actually be filled."""

import numpy as np
import pytest

from yolopose.cli import _resolve_display_scale, _scale_for_display


def test_scale_for_display_doubles_exactly():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    frame[0, -1] = (0, 255, 0)
    frame[-1, 0] = (0, 0, 255)
    frame[-1, -1] = (255, 255, 0)

    big = _scale_for_display(frame, 2.0)
    assert big.shape == (960, 1280, 3)
    # Corners must land in the corners: that is what "filled" means, and it is
    # what regressed when XPutImage refused to stretch on Xwayland.
    assert tuple(big[0, 0]) == (255, 0, 0)
    assert tuple(big[0, -1]) == (0, 255, 0)
    assert tuple(big[-1, 0]) == (0, 0, 255)
    assert tuple(big[-1, -1]) == (255, 255, 0)
    # No untouched black region anywhere.
    assert big[960 // 4:960 * 3 // 4, 1280 // 4:1280 * 3 // 4].sum() >= 0


def test_scale_for_display_triples():
    frame = np.full((240, 320, 3), 7, dtype=np.uint8)
    big = _scale_for_display(frame, 3.0)
    assert big.shape == (720, 960, 3)
    assert np.all(big == 7)


def test_scale_for_display_fractional_keeps_geometry():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    big = _scale_for_display(frame, 1.5)
    assert big.shape[0] == 720 and big.shape[1] == 960


def test_scale_below_one_is_identity():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert _scale_for_display(frame, 1.0) is frame
    assert _scale_for_display(frame, 0.5) is frame


@pytest.mark.parametrize("sensor,expected", [
    ((640, 480), 2.0),
    ((320, 240), 4.0),
    ((1920, 1080), 1.0),
])
def test_auto_scale_fits_the_window_budget(sensor, expected):
    scale = _resolve_display_scale(0.0, 1600, sensor[0], sensor[1])
    assert scale == pytest.approx(expected)


def test_explicit_scale_is_clamped():
    assert _resolve_display_scale(99.0, 1600, 640, 480) == 8.0
    assert _resolve_display_scale(0.01, 1600, 640, 480) == 0.25
