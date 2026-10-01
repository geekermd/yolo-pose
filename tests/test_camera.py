"""Camera tests that do not require capture hardware."""

import numpy as np
import pytest

from yolopose.vision import camera as C
from yolopose.vision.capture import _ffmpeg_pixel_format, camera_backends


def test_ioctl_codes_match_kernel_macros():
    """The computed ioctl numbers must equal the kernel's asm-generic values.

    This is the regression guard for a real bug: the direction field sits at
    bit 30, and shifting it to bit 32 produced codes the kernel rejected with
    ENOTTY.
    """
    assert C.VIDIOC_QUERYCAP == 0x80685600
    assert C.VIDIOC_ENUM_FMT == 0xC0405602
    # v4l2_format is 208 bytes on x86-64 (type + a 204-byte union whose size is
    # driven by v4l2_window), which is why these codes carry size 0xd0.
    assert C.VIDIOC_G_FMT == 0xC0D05604
    assert C.VIDIOC_S_FMT == 0xC0D05605
    assert C.VIDIOC_REQBUFS == 0xC0145608
    assert C.VIDIOC_QUERYBUF == 0xC0585609
    assert C.VIDIOC_QBUF == 0xC058560F
    assert C.VIDIOC_DQBUF == 0xC0585611
    assert C.VIDIOC_STREAMON == 0x40045612
    assert C.VIDIOC_STREAMOFF == 0x40045613


def test_struct_sizes_match_kernel():
    """Kernel-Visible struct sizes; a wrong size is rejected with ENOTTY."""
    import ctypes

    assert ctypes.sizeof(C._V4L2Capability) == 104
    # The union is 200 in this reproduction but the struct still lands on the
    # kernel's 208 because of union alignment; what the kernel checks is the
    # struct size encoded in the ioctl number.
    assert ctypes.sizeof(C._V4L2Window) == 56
    assert ctypes.sizeof(C._V4L2Format) == 208
    assert ctypes.sizeof(C._V4L2Buffer) == 88
    assert ctypes.sizeof(C._V4L2RequestBuffers) == 20
    assert ctypes.sizeof(C._V4L2Fmtdesc) == 64


def test_fourcc_names_prefer_canonical_spelling():
    # JPEG and MJPG share a code; MJPG is the conventional spelling.
    assert C.FOURCC["JPEG"] == C.FOURCC["MJPG"]
    assert C.FOURCC_NAME[C.FOURCC["MJPG"]] == "MJPG"


def test_yuyv_to_rgb_grey_and_red():
    grey = np.full((2, 4, 2), 128, dtype=np.uint8)
    out = C.yuyv_to_rgb(grey)
    assert out.shape == (2, 4, 3)
    assert 100 < out.mean() < 160

    red = np.zeros((2, 4, 2), dtype=np.uint8)
    red[:, 0::2, 0] = 76      # Y
    red[:, 1::2, 0] = 76
    red[:, 0::2, 1] = 84      # U
    red[:, 1::2, 1] = 255     # V
    out = C.yuyv_to_rgb(red)
    assert out[0, 0, 0] > 200 and out[0, 0, 1] < 80 and out[0, 0, 2] < 80


def test_yuyv_requires_even_width():
    with pytest.raises(C.CameraError):
        C.yuyv_to_rgb(np.zeros((2, 3, 2), dtype=np.uint8))


def test_nv12_to_rgb_shape():
    raw = np.full(640 * 480 * 3 // 2, 128, dtype=np.uint8)
    out = C.nv12_to_rgb(raw, 640, 480)
    assert out.shape == (480, 640, 3)


def test_ffmpeg_pixel_format_mapping():
    assert _ffmpeg_pixel_format("MJPG") == "mjpeg"
    assert _ffmpeg_pixel_format("YUYV") == "yuyv422"
    assert _ffmpeg_pixel_format("BGR3") == "bgr24"


def test_camera_backends_lists_v4l2_first():
    backends = camera_backends()
    assert backends[0] == "v4l2"
    assert "ffmpeg" in backends


def test_open_any_camera_rejects_unknown_backend():
    with pytest.raises(ValueError):
        from yolopose.vision.capture import open_any_camera
        open_any_camera("/dev/video0", backend="nonsense")


def test_missing_device_raises():
    with pytest.raises(C.CameraError):
        C.V4L2Camera("/dev/video99", 64, 48, 30, "YUYV")


def test_metadata_device_is_rejected():
    """A UVC metadata node must be reported as not-a-capture-device."""
    import os

    if not os.path.exists("/dev/video1"):
        pytest.skip("no /dev/video1 on this system")
    try:
        camera = C.V4L2Camera("/dev/video1", 64, 48, 30, "YUYV")
    except C.CameraError as exc:
        assert "capture" in str(exc).lower() or "V4L2" in str(exc)
    else:
        camera.close()
        pytest.skip("/dev/video1 is a capture device on this system")
