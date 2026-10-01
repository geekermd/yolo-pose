"""V4L2 camera capture using only the standard library.

The driver's streaming buffers are memory-mapped with :mod:`mmap`, so frames are
read without an extra userspace copy.  Pixel formats ``YUYV``, ``UYVY``,
``RGB24``, ``BGR24``, ``NV12`` and ``MJPG`` are supported; JPEG frames are decoded
by the decoder in :mod:`yolopose.vision.jpeg`.

The ioctl argument structures are declared with :mod:`ctypes` so their sizes and
layouts are exact — ``v4l2_format`` in particular is a tagged union that is far
larger than the pixel-format fields inside it, and using a short buffer makes the
kernel reject the request with ``ENOTTY``.
"""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import select
import struct
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .jpeg import decode_jpeg

# --------------------------------------------------------------------------- #
# ioctl plumbing
# --------------------------------------------------------------------------- #

# Linux asm-generic/ioctl.h layout:
#   bits  0-7  : nr        (8 bits)
#   bits  8-15 : type      (8 bits)
#   bits 16-29 : size      (14 bits)
#   bits 30-31 : direction (2 bits)
# DIR must sit at bit 30; shifting it past bit 31 silently produces a code the
# kernel rejects with ENOTTY (as happened here for VIDIOC_S_FMT).
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = 8
_IOC_SIZESHIFT = 16
_IOC_DIRSHIFT = 30
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2


def _IOC(direction: int, type_: str, nr: int, size: int) -> int:
    return ((direction << _IOC_DIRSHIFT) | (ord(type_) << _IOC_TYPESHIFT)
            | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT))


def _IOR(type_: str, nr: int, size: int) -> int:
    return _IOC(_IOC_READ, type_, nr, size)


def _IOW(type_: str, nr: int, size: int) -> int:
    return _IOC(_IOC_WRITE, type_, nr, size)


def _IOWR(type_: str, nr: int, size: int) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, size)


V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1
V4L2_FIELD_ANY = 0
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_STREAMING = 0x04000000
V4L2_CAP_READWRITE = 0x01000000
V4L2_FMT_FLAG_COMPRESSED = 0x00000010


# --------------------------------------------------------------------------- #
# kernel structure definitions
# --------------------------------------------------------------------------- #

class _V4L2PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _V4L2Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int32), ("top", ctypes.c_int32),
                ("width", ctypes.c_int32), ("height", ctypes.c_int32)]


class _V4L2Clip(ctypes.Structure):
    _fields_ = [("c", _V4L2Rect), ("next", ctypes.c_void_p)]


class _V4L2Window(ctypes.Structure):
    """``v4l2_window``; the trailing flexible array forces 8-byte alignment.

    This member is what makes ``sizeof(struct v4l2_format)`` 208 rather than 204
    on x86-64, so it must be modelled exactly: a 4-byte-short ``v4l2_format``
    fails the kernel's ``_IOC_SIZE`` check and every G_FMT/S_FMT call comes back
    as ENOTTY.
    """

    _fields_ = [
        ("w", _V4L2Rect),
        ("field", ctypes.c_uint32),
        ("chromakey", ctypes.c_uint32),
        ("clips", ctypes.POINTER(_V4L2Clip)),
        ("clipcount", ctypes.c_uint32),
        ("bitmap", ctypes.c_void_p),
        ("global_alpha", ctypes.c_uint8),
    ]


class _V4L2FormatUnion(ctypes.Union):
    """Union of ``v4l2_format`` members; ``win`` is the largest at 56 bytes.

    The kernel's 204-byte union is made up by ``v4l2_window``'s trailing
    flexible array plus its 8-byte alignment, which this reproduction matches, so
    ``sizeof(v4l2_format)`` comes out at the required 208.
    """

    _fields_ = [
        ("pix", _V4L2PixFormat),
        ("win", _V4L2Window),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class _V4L2Format(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("fmt", _V4L2FormatUnion),
    ]


class _V4L2Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _V4L2Fmtdesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _V4L2RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 1),
    ]


class _V4L2Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _V4L2Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _V4L2BufferUnion(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong),
                ("planes", ctypes.c_void_p), ("fd", ctypes.c_int32)]


class _V4L2Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _V4L2Timeval),
        ("timecode", _V4L2Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _V4L2BufferUnion),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 1),
    ]


VIDIOC_QUERYCAP = _IOR("V", 0, ctypes.sizeof(_V4L2Capability))
VIDIOC_ENUM_FMT = _IOWR("V", 2, ctypes.sizeof(_V4L2Fmtdesc))
VIDIOC_G_FMT = _IOWR("V", 4, ctypes.sizeof(_V4L2Format))
VIDIOC_S_FMT = _IOWR("V", 5, ctypes.sizeof(_V4L2Format))
VIDIOC_REQBUFS = _IOWR("V", 8, ctypes.sizeof(_V4L2RequestBuffers))
VIDIOC_QUERYBUF = _IOWR("V", 9, ctypes.sizeof(_V4L2Buffer))
VIDIOC_QBUF = _IOWR("V", 15, ctypes.sizeof(_V4L2Buffer))
VIDIOC_DQBUF = _IOWR("V", 17, ctypes.sizeof(_V4L2Buffer))
VIDIOC_STREAMON = _IOW("V", 18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _IOW("V", 19, ctypes.sizeof(ctypes.c_int))

FOURCC = {
    "YUYV": 0x56595559, "YUY2": 0x56595559,
    "UYVY": 0x59565955,
    "MJPG": 0x47504A4D, "JPEG": 0x47504A4D,
    "RGB3": 0x33424752, "RGB24": 0x33424752,
    "BGR3": 0x33475242, "BGR24": 0x33475242,
    "GREY": 0x59455247, "NV12": 0x3231564E,
}
# Several aliases map to the same code; keep the canonical name for display.
_CANONICAL = ("YUYV", "UYVY", "MJPG", "RGB3", "BGR3", "GREY", "NV12")
FOURCC_NAME: Dict[int, str] = {}
for _name in _CANONICAL:
    FOURCC_NAME.setdefault(FOURCC[_name], _name)
for _name, _code in FOURCC.items():
    FOURCC_NAME.setdefault(_code, _name)


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or configured."""


@dataclass
class CameraFormat:
    """Negotiated capture format."""

    fourcc: str
    width: int
    height: int
    size_image: int = 0
    field: int = V4L2_FIELD_NONE
    bytes_per_line: int = 0

    def __str__(self) -> str:
        return f"{self.fourcc} {self.width}x{self.height}"


def _size_candidates(width: int, height: int) -> List[Tuple[int, int]]:
    """Requested size first, then smaller standard modes."""
    candidates: List[Tuple[int, int]] = []
    if width > 0 and height > 0:
        candidates.append((int(width), int(height)))
    for mode in ((640, 480), (320, 240), (1280, 720), (1920, 1080)):
        if mode not in candidates:
            candidates.append(mode)
    return candidates


def _close_geometry(actual_w: int, actual_h: int, want_w: int, want_h: int,
                    tolerance: float = 1.25) -> bool:
    """True when the driver stayed within ``tolerance`` of the requested size."""
    if want_w <= 0 or want_h <= 0:
        return True
    if actual_w <= 0 or actual_h <= 0:
        return False
    return (actual_w <= want_w * tolerance) and (actual_h <= want_h * tolerance)


def _decode_c_string(raw: bytes) -> str:
    return raw.split(b"\x00")[0].decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #

class V4L2Camera:
    """A single V4L2 capture device with mmap streaming.

    Use as a context manager::

        with V4L2Camera(0, width=640, height=480) as camera:
            frame = camera.read()
    """

    def __init__(self, device="0", width: int = 640, height: int = 480,
                 fps: int = 30, pixel_format: str = "YUYV", buffers: int = 4,
                 timeout: float = 2.0):
        if isinstance(device, int):
            device = f"/dev/video{device}"
        elif isinstance(device, str) and device.isdigit():
            device = f"/dev/video{device}"
        self.device = device
        self.timeout = float(timeout)
        self.requested = (pixel_format, width, height, fps)

        if not os.path.exists(device):
            raise CameraError(
                f"{device} does not exist; pass an explicit device such as /dev/video0")

        try:
            self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        except OSError as exc:
            raise CameraError(f"cannot open {device}: {exc}") from exc

        self.buffers: List[mmap.mmap] = []
        self.buffer_lengths: List[int] = []
        self.buffer_count = 0
        self.format: Optional[CameraFormat] = None
        self._streaming = False
        self._closed = False
        self._capabilities = 0
        self._device_caps = 0

        try:
            self._query_capabilities()
            self._set_format(pixel_format, width, height)
            self._request_buffers(buffers)
            self._map_buffers()
            self._queue_all()
        except Exception:
            self.close()
            raise

    # -- capabilities ------------------------------------------------------- #
    def _query_capabilities(self) -> None:
        capability = _V4L2Capability()
        try:
            fcntl.ioctl(self.fd, VIDIOC_QUERYCAP, capability, True)
        except OSError as exc:
            raise CameraError(
                f"{self.device} is not a V4L2 device ({exc}); it may be a "
                "metadata-only node such as /dev/video1 on UVC webcams"
            ) from exc
        self._capabilities = capability.capabilities
        self._device_caps = capability.device_caps or capability.capabilities

        if not (self._device_caps & V4L2_CAP_VIDEO_CAPTURE):
            raise CameraError(
                f"{self.device} is not a video capture device "
                f"(device_caps=0x{self._device_caps:08x}); try another index"
            )

    def capabilities(self) -> Dict[str, object]:
        """Driver name, card name and capability flags."""
        capability = _V4L2Capability()
        fcntl.ioctl(self.fd, VIDIOC_QUERYCAP, capability, True)
        return {
            "driver": _decode_c_string(capability.driver),
            "card": _decode_c_string(capability.card),
            "bus": _decode_c_string(capability.bus_info),
            "version": capability.version,
            "capabilities": capability.capabilities,
            "device_caps": capability.device_caps,
            "capture": bool(self._device_caps & V4L2_CAP_VIDEO_CAPTURE),
            "streaming": bool(self._device_caps & V4L2_CAP_STREAMING),
            "readwrite": bool(self._device_caps & V4L2_CAP_READWRITE),
        }

    def enumerate_formats(self) -> List[Tuple[str, str, bool]]:
        """``(fourcc, description, compressed)`` for each offered format."""
        found: List[Tuple[str, str, bool]] = []
        for index in range(32):
            description = _V4L2Fmtdesc()
            description.index = index
            description.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            try:
                fcntl.ioctl(self.fd, VIDIOC_ENUM_FMT, description, True)
            except OSError:
                break
            fourcc = FOURCC_NAME.get(description.pixelformat,
                                     f"0x{description.pixelformat:08x}")
            found.append((fourcc, _decode_c_string(description.description),
                          bool(description.flags & V4L2_FMT_FLAG_COMPRESSED)))
        return found

    # -- format ------------------------------------------------------------- #
    def _set_format(self, pixel_format: str, width: int, height: int) -> None:
        """Negotiate a capture format, honouring the requested size if possible.

        The driver is allowed to pick its own geometry, so the result is checked:
        a camera that answers a 640x480 request with 1920x1080 would otherwise
        silently consume far more USB bandwidth than asked for.
        """
        if pixel_format.upper() not in FOURCC:
            raise CameraError(
                f"unknown pixel format {pixel_format!r}; choose from "
                f"{', '.join(sorted(set(FOURCC)))}")

        errors: List[str] = []
        # Pass 1: the requested size, then smaller conventional modes.
        for attempt_w, attempt_h in _size_candidates(width, height):
            try:
                self._apply_format(pixel_format.upper(), attempt_w, attempt_h)
            except OSError as exc:
                errors.append(f"{attempt_w}x{attempt_h}: {exc}")
                continue
            # Accept only if the driver kept the scale we asked for.
            if _close_geometry(self.format.width, self.format.height,
                               attempt_w, attempt_h):
                return
            errors.append(
                f"{attempt_w}x{attempt_h}: driver answered "
                f"{self.format.width}x{self.format.height}")

        # Pass 2: give the driver free choice (width/height of 0 means "any").
        try:
            self._apply_format(pixel_format.upper(), 0, 0)
            if self.format.width > 0 and self.format.height > 0:
                return
        except OSError as exc:
            errors.append(f"driver default: {exc}")

        raise CameraError(
            f"could not negotiate {pixel_format} ({'; '.join(errors[:3])})")

    def _apply_format(self, pixel_format: str, width: int, height: int) -> None:
        fourcc = FOURCC[pixel_format]
        fmt = _V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = int(width)
        fmt.fmt.pix.height = int(height)
        fmt.fmt.pix.pixelformat = fourcc
        fmt.fmt.pix.field = V4L2_FIELD_ANY
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt, True)

        actual_fourcc = fmt.fmt.pix.pixelformat
        if actual_fourcc != fourcc:
            raise CameraError(
                "driver selected "
                f"{FOURCC_NAME.get(actual_fourcc, hex(actual_fourcc))} instead of "
                f"{pixel_format}")

        self.format = CameraFormat(
            fourcc=FOURCC_NAME.get(actual_fourcc, hex(actual_fourcc)),
            width=int(fmt.fmt.pix.width),
            height=int(fmt.fmt.pix.height),
            size_image=int(fmt.fmt.pix.sizeimage),
            field=int(fmt.fmt.pix.field),
            bytes_per_line=int(fmt.fmt.pix.bytesperline),
        )
        if self.format.size_image <= 0:
            raise CameraError("driver reported a zero-sized image buffer")

    # -- buffers ------------------------------------------------------------ #
    def _request_buffers(self, count: int) -> None:
        request = _V4L2RequestBuffers()
        request.count = int(count)
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        request.memory = V4L2_MEMORY_MMAP
        try:
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, request, True)
        except OSError as exc:
            raise CameraError(
                f"VIDIOC_REQBUFS failed ({exc}); the device may not support "
                "memory-mapped streaming") from exc
        if request.count < 2:
            raise CameraError(
                f"driver granted only {request.count} buffer(s); mmap streaming "
                "needs at least 2")
        self.buffer_count = int(request.count)

    def _map_buffers(self) -> None:
        for index in range(self.buffer_count):
            buffer = _V4L2Buffer()
            buffer.index = index
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buffer.memory = V4L2_MEMORY_MMAP
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buffer, True)
            length = int(buffer.length) or (
                self.format.size_image if self.format else 0)
            if length <= 0:
                raise CameraError(f"buffer {index} reports zero length")
            mapped = mmap.mmap(self.fd, length, mmap.MAP_SHARED,
                               mmap.PROT_READ | mmap.PROT_WRITE,
                               offset=int(buffer.m.offset))
            self.buffers.append(mapped)
            self.buffer_lengths.append(length)

    def _queue(self, index: int) -> None:
        buffer = _V4L2Buffer()
        buffer.index = index
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buffer.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buffer, True)

    def _queue_all(self) -> None:
        for index in range(self.buffer_count):
            self._queue(index)

    # -- streaming ---------------------------------------------------------- #
    def start(self) -> None:
        if self._streaming:
            return
        fcntl.ioctl(self.fd, VIDIOC_STREAMON,
                    struct.pack("i", V4L2_BUF_TYPE_VIDEO_CAPTURE))
        self._streaming = True

    def stop(self) -> None:
        if not self._streaming:
            return
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF,
                        struct.pack("i", V4L2_BUF_TYPE_VIDEO_CAPTURE))
        finally:
            self._streaming = False

    def read(self, raw: bool = False) -> np.ndarray:
        """Grab one frame as an ``(H, W, 3)`` RGB ``uint8`` array.

        With ``raw=True`` the untouched driver payload is returned instead, which
        is useful when debugging pixel formats.
        """
        if self._closed:
            raise CameraError("camera is closed")
        if not self._streaming:
            self.start()
        if self.format is None:
            raise CameraError("camera has no negotiated format")

        ready, _, _ = select.select([self.fd], [], [], self.timeout)
        if not ready:
            raise CameraError(
                f"timed out after {self.timeout:.1f}s waiting for a frame; the "
                "device may be in use by another process")

        buffer = _V4L2Buffer()
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buffer.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, buffer, True)
        index = int(buffer.index)
        used = int(buffer.bytesused) or self.buffer_lengths[index]
        try:
            payload = self.buffers[index][:used]
            if raw:
                return np.frombuffer(payload, dtype=np.uint8).copy()
            return self._to_rgb(payload)
        finally:
            self._queue(index)

    def _to_rgb(self, payload) -> np.ndarray:
        fmt = self.format
        assert fmt is not None
        expected = fmt.width * fmt.height
        fourcc = fmt.fourcc.upper()

        if fourcc in ("MJPG", "JPEG"):
            return decode_jpeg(bytes(payload))
        if fourcc in ("YUYV", "YUY2"):
            raw = np.frombuffer(payload, dtype=np.uint8, count=expected * 2)
            return yuyv_to_rgb(raw.reshape(fmt.height, fmt.width, 2))
        if fourcc == "UYVY":
            raw = np.frombuffer(payload, dtype=np.uint8, count=expected * 2)
            return yuyv_to_rgb(raw.reshape(fmt.height, fmt.width, 2)[:, :, ::-1])
        if fourcc in ("RGB3", "RGB24"):
            return np.frombuffer(payload, dtype=np.uint8,
                                 count=expected * 3).reshape(
                fmt.height, fmt.width, 3).copy()
        if fourcc in ("BGR3", "BGR24"):
            raw = np.frombuffer(payload, dtype=np.uint8,
                                count=expected * 3).reshape(fmt.height, fmt.width, 3)
            return np.ascontiguousarray(raw[:, :, ::-1])
        if fourcc == "GREY":
            raw = np.frombuffer(payload, dtype=np.uint8,
                                count=expected).reshape(fmt.height, fmt.width)
            return np.repeat(raw[:, :, None], 3, axis=2)
        if fourcc == "NV12":
            return nv12_to_rgb(
                np.frombuffer(payload, dtype=np.uint8, count=expected * 3 // 2),
                fmt.width, fmt.height)
        raise CameraError(f"unsupported capture format {fmt.fourcc}")

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
        except Exception:
            pass
        for mapped in self.buffers:
            try:
                mapped.close()
            except Exception:
                pass
        self.buffers = []
        try:
            os.close(self.fd)
        except Exception:
            pass

    def __enter__(self) -> "V4L2Camera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        while not self._closed:
            yield self.read()


# --------------------------------------------------------------------------- #
# pixel conversions
# --------------------------------------------------------------------------- #

def yuyv_to_rgb(raw: np.ndarray) -> np.ndarray:
    """Convert packed YUYV (YUY2) to RGB with integer arithmetic.

    Byte order per 4-byte group is ``Y0 U Y1 V``: one chroma pair is shared by two
    horizontally adjacent luma samples.
    """
    height, width = raw.shape[:2]
    if width % 2:
        raise CameraError("YUYV frames must have an even width")
    groups = raw.reshape(height, width // 2, 2, 2)
    y0 = groups[:, :, 0, 0].astype(np.int32)
    u_plane = groups[:, :, 0, 1].astype(np.int32)
    y1 = groups[:, :, 1, 0].astype(np.int32)
    v_plane = groups[:, :, 1, 1].astype(np.int32)

    y_full = np.empty((height, width), dtype=np.int32)
    y_full[:, 0::2] = y0
    y_full[:, 1::2] = y1
    u_full = u_plane.repeat(2, axis=1) - 128
    v_full = v_plane.repeat(2, axis=1) - 128

    # BT.601 studio-swing YUV -> RGB, integer coefficients scaled by 256.
    y_scaled = np.maximum(y_full - 16, 0) * 298
    r = (y_scaled + 409 * v_full + 128) >> 8
    g = (y_scaled - 100 * u_full - 208 * v_full + 128) >> 8
    b = (y_scaled + 516 * u_full + 128) >> 8
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


def nv12_to_rgb(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert an NV12 (semi-planar) frame to RGB."""
    luma = raw[:width * height].reshape(height, width).astype(np.int32)
    chroma = raw[width * height:].reshape(height // 2, width // 2, 2).astype(np.int32)
    cb = np.repeat(np.repeat(chroma[:, :, 0], 2, axis=0), 2, axis=1)[:height, :width] - 128
    cr = np.repeat(np.repeat(chroma[:, :, 1], 2, axis=0), 2, axis=1)[:height, :width] - 128
    y_scaled = np.maximum(luma - 16, 0) * 298
    r = (y_scaled + 409 * cr + 128) >> 8
    g = (y_scaled - 100 * cb - 208 * cr + 128) >> 8
    b = (y_scaled + 516 * cb + 128) >> 8
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def list_cameras() -> List[str]:
    """Paths of every ``/dev/video*`` node."""
    return [f"/dev/video{index}" for index in range(16)
            if os.path.exists(f"/dev/video{index}")]


def capture_devices() -> List[Tuple[str, Dict[str, object]]]:
    """Probe every video node, returning only true capture devices."""
    found: List[Tuple[str, Dict[str, object]]] = []
    for path in list_cameras():
        try:
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            capability = _V4L2Capability()
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, capability, True)
            device_caps = capability.device_caps or capability.capabilities
            if not device_caps & V4L2_CAP_VIDEO_CAPTURE:
                continue
            formats: List[str] = []
            for index in range(16):
                description = _V4L2Fmtdesc()
                description.index = index
                description.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FMT, description, True)
                except OSError:
                    break
                formats.append(FOURCC_NAME.get(description.pixelformat,
                                               f"0x{description.pixelformat:08x}"))
            found.append((path, {
                "card": _decode_c_string(capability.card),
                "driver": _decode_c_string(capability.driver),
                "formats": formats,
            }))
        except OSError:
            continue
        finally:
            os.close(fd)
    return found


def open_camera(device="0", width: int = 640, height: int = 480, fps: int = 30,
                pixel_format: str = "auto", timeout: float = 2.0) -> V4L2Camera:
    """Open the first format that works.

    ``pixel_format='auto'`` prefers YUYV (uncompressed, no decoder needed) and
    falls back to MJPEG, which is what most UVC webcams stream at high
    resolutions.
    """
    if pixel_format.lower() in ("auto", ""):
        candidates = ["YUYV", "MJPG", "RGB3", "BGR3", "NV12"]
    else:
        candidates = [pixel_format.upper(), "YUYV", "MJPG", "RGB3", "BGR3"]

    seen = set()
    errors: List[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return V4L2Camera(device, width, height, fps, candidate,
                              timeout=timeout)
        except CameraError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
    detail = "; ".join(errors[:4]) if errors else "no formats attempted"
    raise CameraError(f"could not open {device}: {detail}")
