"""Image loading, saving and letterbox resizing without native dependencies.

Image *decoding* (PNG/JPEG/BMP) prefers Pillow when the distribution provides
it and otherwise falls back to a built-in pure-Python PPM/PGM/PNM reader plus a
font-free PNG writer, so the tool still runs on a machine that only has NumPy.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # Pillow is a soft dependency (python3-pil on Debian/Ubuntu)
    from PIL import Image as _PILImage
    HAVE_PIL = True
except Exception:  # pragma: no cover - exercised on minimal installs
    _PILImage = None
    HAVE_PIL = False


class ImageError(RuntimeError):
    """Raised when an image cannot be read or written."""


@dataclass
class LetterboxInfo:
    """Geometry needed to map network coordinates back to the source image."""

    scale: float
    pad_x: float
    pad_y: float
    src_width: int
    src_height: int
    net_width: int
    net_height: int


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_image(path: str) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array of shape ``(H, W, 3)``."""
    if not os.path.exists(path):
        raise ImageError(f"no such image: {path}")

    with open(path, "rb") as handle:
        head = handle.read(2)

    if head == b"P6" or head == b"P5" or head == b"P3" or head == b"P2":
        return _load_pnm(path)
    if head == b"\x89P":
        return _load_png(path)

    if HAVE_PIL:
        with _PILImage.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    raise ImageError(
        f"cannot decode {path!r}: install python3-pil for JPEG/BMP/PNG support"
    )


def _read_pnm_tokens(handle, count: int = 3) -> list:
    """Read exactly ``count`` whitespace-separated header tokens.

    A P6/P5 header carries three values (width, height, maxval).  Reading a
    fourth token would swallow the first byte of pixel data and truncate the
    image.
    """
    tokens: list = []
    while len(tokens) < count:
        line = handle.readline()
        if not line:
            break
        if line.startswith(b"#"):
            continue
        tokens.extend(line.split())
    return tokens[:count]


def _load_pnm(path: str) -> np.ndarray:
    """Minimal binary/ASCII PPM (P6/P3) and PGM (P5/P2) reader."""
    with open(path, "rb") as handle:
        magic = handle.read(2)
        tokens = _read_pnm_tokens(handle)
        if len(tokens) < 3:
            raise ImageError(f"malformed PNM header in {path!r}")
        width, height, maxval = (int(t) for t in tokens[:3])
        if maxval > 255:
            raise ImageError("16-bit PNM images are not supported")

        if magic in (b"P6", b"P5"):
            channels = 3 if magic == b"P6" else 1
            raw = handle.read(width * height * channels)
            array = np.frombuffer(raw, dtype=np.uint8)
            if array.size != width * height * channels:
                raise ImageError(f"truncated PNM data in {path!r}")
            image = array.reshape(height, width, channels)
        else:  # ASCII variants
            values = np.fromiter(
                (int(v) for v in handle.read().split()), dtype=np.uint16,
                count=width * height * (3 if magic == b"P3" else 1))
            channels = 3 if magic == b"P3" else 1
            image = values.reshape(height, width, channels).astype(np.uint8)

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(image)


def _load_png(path: str) -> np.ndarray:
    """Minimal PNG reader (8-bit, non-interlaced, colour types 0/2/3/4/6)."""
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageError(f"{path!r} is not a PNG file")

    pos = 8
    width = height = bit_depth = colour_type = interlace = 0
    palette: Optional[np.ndarray] = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length, chunk_type = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, colour_type, _comp, _filt, interlace = \
                struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"PLTE":
            palette = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 3)
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8 or interlace != 0:
        if HAVE_PIL:
            with _PILImage.open(path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        raise ImageError("only 8-bit non-interlaced PNG is supported without Pillow")

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colour_type)
    if channels is None:
        raise ImageError(f"unsupported PNG colour type {colour_type}")

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = np.zeros((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        line = np.frombuffer(raw[offset:offset + stride], dtype=np.uint8).astype(np.int16)
        offset += stride
        if filter_type == 0:
            current = line
        elif filter_type == 1:  # Sub
            current = line.copy()
            for i in range(channels, stride):
                current[i] = (current[i] + current[i - channels]) & 0xFF
        elif filter_type == 2:  # Up
            current = (line + previous) & 0xFF
        elif filter_type == 3:  # Average
            current = line.copy()
            for i in range(stride):
                left = current[i - channels] if i >= channels else 0
                current[i] = (current[i] + ((int(left) + int(previous[i])) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            current = line.copy()
            for i in range(stride):
                a = int(current[i - channels]) if i >= channels else 0
                b = int(previous[i])
                c = int(previous[i - channels]) if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                current[i] = (current[i] + pred) & 0xFF
        else:
            raise ImageError(f"unknown PNG filter {filter_type}")
        out[row] = current.astype(np.uint8)
        previous = out[row]

    image = out.reshape(height, width, channels)
    if colour_type == 3:
        if palette is None:
            raise ImageError("palette PNG without PLTE chunk")
        image = palette[image[:, :, 0]]
    elif colour_type == 0:
        image = np.repeat(image, 3, axis=2)
    elif colour_type == 4:
        image = np.repeat(image[:, :, :1], 3, axis=2)
    elif colour_type == 6:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


# --------------------------------------------------------------------------- #
# saving
# --------------------------------------------------------------------------- #

def save_image(path: str, image: np.ndarray, quality: int = 92) -> None:
    """Write an RGB array to disk, choosing the format from the extension."""
    suffix = os.path.splitext(path)[1].lower()
    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)

    if suffix in (".ppm", ".pnm", ".pgm"):
        _save_pnm(path, array)
        return
    if suffix == ".png" or not suffix:
        _save_png(path, array)
        return
    if HAVE_PIL:
        with _PILImage.fromarray(array) as pil_image:
            pil_image.save(path, quality=quality)
        return
    # Fall back to PNG bytes with a corrected extension hint.
    _save_png(path, array)


def _save_pnm(path: str, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    payload = image[:, :, :3].tobytes() if image.ndim == 3 else image.tobytes()
    with open(path, "wb") as handle:
        handle.write(b"P6\n%d %d\n255\n" % (width, height))
        handle.write(payload)


def _save_png(path: str, image: np.ndarray) -> None:
    """Write an 8-bit RGB PNG using only the standard library."""
    height, width = image.shape[:2]
    rgb = np.ascontiguousarray(image[:, :, :3])

    raw = bytearray()
    stride = width * 3
    for row in range(height):
        raw.append(0)  # filter type 0 (None): simplest and always valid
        raw.extend(rgb[row].tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", header))
        handle.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        handle.write(chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def letterbox(image: np.ndarray, size: Tuple[int, int] = (640, 640),
              colour: int = 114) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize preserving aspect ratio and pad to ``size`` (YOLO convention).

    Returns the padded image plus the transform needed to map predictions back.
    """
    net_w, net_h = int(size[0]), int(size[1])
    src_h, src_w = image.shape[:2]
    scale = min(net_w / src_w, net_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = resize_bilinear(image, (new_h, new_w))
    canvas = np.full((net_h, net_w, 3), colour, dtype=np.uint8)
    pad_x = (net_w - new_w) // 2
    pad_y = (net_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    info = LetterboxInfo(scale=scale, pad_x=float(pad_x), pad_y=float(pad_y),
                         src_width=src_w, src_height=src_h,
                         net_width=net_w, net_height=net_h)
    return canvas, info


def resize_bilinear(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize to ``(height, width)`` using NumPy only."""
    out_h, out_w = int(size[0]), int(size[1])
    in_h, in_w = image.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return np.ascontiguousarray(image)

    src = image.astype(np.float32, copy=False)
    # Align pixel centres (same convention as OpenCV's INTER_LINEAR).
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    y = np.clip(y, 0, in_h - 1)
    x = np.clip(x, 0, in_w - 1)

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (y - y0).reshape(-1, 1, 1)
    wx = (x - x0).reshape(1, -1, 1)

    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    out = top * (1 - wy) + bottom * wy
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def to_network_input(image: np.ndarray) -> np.ndarray:
    """RGB HWC uint8 -> normalised float32 NCHW tensor in ``[0, 1]``."""
    tensor = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def map_point(x: float, y: float, info: LetterboxInfo) -> Tuple[float, float]:
    """Map a network-space point back to source-image coordinates."""
    return ((x - info.pad_x) / info.scale, (y - info.pad_y) / info.scale)


def map_box(box: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map ``[x1, y1, x2, y2]`` from network space to source coordinates."""
    out = box.astype(np.float64).copy()
    out[..., 0] = (box[..., 0] - info.pad_x) / info.scale
    out[..., 2] = (box[..., 2] - info.pad_x) / info.scale
    out[..., 1] = (box[..., 1] - info.pad_y) / info.scale
    out[..., 3] = (box[..., 3] - info.pad_y) / info.scale
    out[..., 0] = np.clip(out[..., 0], 0, info.src_width - 1)
    out[..., 2] = np.clip(out[..., 2], 0, info.src_width - 1)
    out[..., 1] = np.clip(out[..., 1], 0, info.src_height - 1)
    out[..., 3] = np.clip(out[..., 3], 0, info.src_height - 1)
    return out
