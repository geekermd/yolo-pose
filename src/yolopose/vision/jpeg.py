"""JPEG decoding for MJPEG camera streams.

Two code paths exist:

* :func:`decode_jpeg` prefers Pillow (``python3-pil``), which is present on
  every normal Debian/Ubuntu install and is battle-tested.
* :func:`decode_jpeg_builtin` is a dependency-free baseline JPEG decoder kept
  for minimal systems.  It handles single-component and simple baseline
  streams; treat it as **experimental** for arbitrary colour JPEG.

Neither path supports progressive JPEG.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class JpegError(RuntimeError):
    """Raised when the bitstream cannot be decoded."""


# Zig-zag order used by the DQT / DCT coefficient layout.
ZIGZAG = np.asarray([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int32)

# Precomputed 8x8 inverse-DCT basis: basis[u, v, x, y].
def _build_idct_basis() -> np.ndarray:
    x = np.arange(8)
    u = np.arange(8)
    cu = np.where(u == 0, 1.0 / np.sqrt(2.0), 1.0)
    basis = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for uu in range(8):
        for vv in range(8):
            outer = (np.cos((2 * x + 1) * uu * np.pi / 16)[:, None]
                     * np.cos((2 * x + 1) * vv * np.pi / 16)[None, :])
            basis[uu, vv] = 0.25 * cu[uu] * cu[vv] * outer
    return basis


_IDCT_BASIS = _build_idct_basis()


class _BitReader:
    """MSB-first bit reader with 0xFF00 byte-unstuffing.

    Bytes are held in a list so the accumulator cannot grow without bound; the
    next bit is simply the high bit of the current byte.
    """

    def __init__(self, data: bytes, pos: int):
        self.data = data
        self.pos = pos
        self.cur = 0
        self.nbits = 0
        self.eof = False
        self.bit_positions: List[int] = []

    def _next_byte(self) -> Optional[int]:
        while self.pos < len(self.data):
            byte = self.data[self.pos]
            self.pos += 1
            if byte != 0xFF:
                return byte
            if self.pos >= len(self.data):
                self.eof = True
                return None
            marker = self.data[self.pos]
            if marker == 0x00:
                self.pos += 1
                return 0xFF
            if 0xD0 <= marker <= 0xD7:
                # Restart marker: byte-align and reset the DC predictors.
                self.pos += 1
                self.cur = 0
                self.nbits = 0
                continue
            self.eof = True
            return None
        self.eof = True
        return None

    def bit(self) -> int:
        if self.nbits == 0:
            byte = self._next_byte()
            if byte is None:
                return 0
            self.cur = byte
            self.nbits = 8
        self.nbits -= 1
        return (self.cur >> self.nbits) & 1

    def bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value

    def align(self) -> None:
        self.cur = 0
        self.nbits = 0

    @property
    def bit_position(self) -> int:
        return self.pos * 8 - self.nbits


def _extend(value: int, bits: int) -> int:
    """Sign-extend a JPEG magnitude-category value."""
    if bits == 0:
        return 0
    if value < (1 << (bits - 1)):
        return value - (1 << bits) + 1
    return value


class _HuffmanTable:
    """Canonical JPEG Huffman decoder.

    ``counts`` is the raw 16-entry table from a DHT marker: ``counts[i]`` is how
    many symbols are encoded with a code of length ``i + 1``.  Codes are the
    canonical codes of ITU-T T.81 Annex C, built with the standard procedure
    (within a length codes increment; moving to the next length shifts left).
    """

    def __init__(self, counts: List[int], values: List[int]):
        if len(counts) != 16:
            raise JpegError(f"Huffman table must declare 16 code lengths, got {len(counts)}")
        self.counts = list(counts)
        self.values = list(values)
        total = sum(self.counts)
        if total != len(self.values):
            raise JpegError(
                f"Huffman table declares {total} symbols but carries {len(self.values)}"
            )

        codes: dict = {}
        code = 0
        index = 0
        for slot, count in enumerate(self.counts):
            length = slot + 1
            for _ in range(count):
                codes[(length, code)] = self.values[index]
                index += 1
                code += 1
            code <<= 1
        self.codes = codes
        self.max_length = max((l for l, _ in codes), default=0)

        # Fast path for codes that fit in 8 bits.
        self.lookup = {key: value for key, value in codes.items() if key[0] <= 8}

    def decode(self, reader: "_BitReader") -> int:
        code = 0
        for length in range(1, self.max_length + 1):
            code = (code << 1) | reader.bit()
            value = self.lookup.get((length, code))
            if value is not None:
                return value
        raise JpegError("invalid Huffman code")


class _Component:
    __slots__ = ("identifier", "h", "v", "quant", "dc_table", "ac_table",
                 "dc_pred", "blocks", "quant_table", "block_w", "block_h")

    def __init__(self, identifier: int, h: int, v: int, quant: int):
        self.identifier = identifier
        self.h = h
        self.v = v
        self.quant = quant
        self.dc_table: Optional[_HuffmanTable] = None
        self.ac_table: Optional[_HuffmanTable] = None
        self.dc_pred = 0


def _read_huffman_tables(data: bytes, offset: int, length: int) -> dict:
    tables = {}
    end = offset + length
    while offset < end:
        table_id = data[offset]
        offset += 1
        counts = list(data[offset:offset + 16])
        offset += 16
        total = sum(counts)
        symbols = list(data[offset:offset + total])
        offset += total
        table_class = table_id >> 4          # 0 = DC, 1 = AC
        table_slot = table_id & 0x0F
        if table_class not in (0, 1):
            raise JpegError(f"unsupported Huffman table class {table_class}")
        tables[(table_class, table_slot)] = _HuffmanTable(counts, symbols)
    return tables


def _idct(coefficients: np.ndarray) -> np.ndarray:
    """8x8 inverse DCT via the precomputed basis tensor."""
    return np.einsum("uv,uvxy->xy", coefficients.astype(np.float32), _IDCT_BASIS,
                     optimize=True)


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode a JPEG byte string into an ``(H, W, 3)`` RGB array.

    Uses Pillow when available (recommended); otherwise falls back to the
    built-in pure-Python decoder.
    """
    try:
        from .imaging import HAVE_PIL
    except Exception:  # pragma: no cover - circular-import safety
        HAVE_PIL = False

    if HAVE_PIL:
        try:
            import io

            from PIL import Image as _Image
            with _Image.open(io.BytesIO(data)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception:
            pass  # fall through to the built-in decoder
    return decode_jpeg_builtin(data)


def decode_jpeg_builtin(data: bytes) -> np.ndarray:
    """Dependency-free baseline JPEG decoder (single-component streams are the
    best-tested path; multi-component colour streams are experimental)."""
    if data[:2] != b"\xff\xd8":
        raise JpegError("not a JPEG stream (missing SOI marker)")

    quant_tables: dict = {}
    huffman_tables: dict = {}
    components: List[_Component] = []
    width = height = 0
    restart_interval = 0
    offset = 2
    scan_started = False
    adobe_transform: Optional[int] = None
    component_ids: List[int] = []

    while offset < len(data) - 1:
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:  # EOI
            break
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        segment = data[offset + 2:offset + segment_length]
        offset += segment_length

        if marker == 0xDB:  # DQT
            position = 0
            while position < len(segment):
                table_id = segment[position]
                precision = table_id >> 4
                table_index = table_id & 0x0F
                position += 1
                if precision == 0:
                    table = np.frombuffer(segment[position:position + 64],
                                          dtype=np.uint8).astype(np.int32)
                    position += 64
                else:
                    table = np.frombuffer(segment[position:position + 128],
                                          dtype=">u2").astype(np.int32)
                    position += 128
                natural = np.zeros(64, dtype=np.int32)
                natural[ZIGZAG] = table
                quant_tables[table_index] = natural.reshape(8, 8)

        elif marker == 0xC0 or marker == 0xC1:  # SOF0 / SOF1
            precision = segment[0]
            if precision != 8:
                raise JpegError(f"only 8-bit JPEG is supported (got {precision})")
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
            count = segment[5]
            components = []
            for i in range(count):
                base = 6 + i * 3
                identifier = segment[base]
                sampling = segment[base + 1]
                quant_id = segment[base + 2]
                components.append(_Component(identifier, sampling >> 4,
                                             sampling & 0x0F, quant_id))
                component_ids.append(identifier)

        elif marker == 0xC2:
            raise JpegError("progressive JPEG is not supported; use a baseline stream")

        elif marker == 0xC4:  # DHT
            huffman_tables.update(_read_huffman_tables(segment, 0, len(segment)))

        elif marker == 0xDD:  # DRI
            restart_interval = int.from_bytes(segment[0:2], "big")

        elif marker == 0xEE:  # APP14 (Adobe)
            if segment[:5] == b"Adobe":
                adobe_transform = segment[11]

        elif marker == 0xDA:  # SOS
            scan_started = True
            count = segment[0]
            scan_components = []
            for i in range(count):
                base = 1 + i * 2
                identifier = segment[base]
                tables = segment[base + 1]
                component = next((c for c in components if c.identifier == identifier),
                                 None)
                if component is None:
                    raise JpegError(f"scan references unknown component {identifier}")
                component.dc_table = huffman_tables.get((0, tables >> 4))
                component.ac_table = huffman_tables.get((1, tables & 0x0F))
                if component.dc_table is None or component.ac_table is None:
                    raise JpegError("scan references an undefined Huffman table")
                scan_components.append(component)

            for component in scan_components:
                table = quant_tables.get(component.quant)
                if table is None:
                    raise JpegError(
                        f"component {component.identifier} references missing "
                        f"quantisation table {component.quant}"
                    )
                component.quant_table = table

            spectral_start = segment[1 + count * 2]
            spectral_end = segment[2 + count * 2]
            if spectral_start != 0 or spectral_end != 63:
                raise JpegError("sequential JPEG with spectral selection is unsupported")

            _decode_scan(data, offset, width, height, scan_components,
                         restart_interval, adobe_transform)
            break

    if not scan_started:
        raise JpegError("JPEG contains no scan data")
    return _assemble(components, width, height, adobe_transform, len(component_ids))


def _decode_scan(data: bytes, offset: int, width: int, height: int,
                 components: List[_Component], restart_interval: int,
                 adobe_transform: Optional[int]) -> None:
    max_h = max(c.h for c in components)
    max_v = max(c.v for c in components)
    mcu_width = 8 * max_h
    mcu_height = 8 * max_v
    mcus_x = (width + mcu_width - 1) // mcu_width
    mcus_y = (height + mcu_height - 1) // mcu_height

    for component in components:
        component.block_w = mcus_x * component.h
        component.block_h = mcus_y * component.v
        component.blocks = np.zeros((component.block_h, component.block_w, 64),
                                    dtype=np.float32)
        component.dc_pred = 0

    reader = _BitReader(data, offset)
    mcu_count = 0
    for mcu_y in range(mcus_y):
        for mcu_x in range(mcus_x):
            if restart_interval and mcu_count and mcu_count % restart_interval == 0:
                reader.align()
                for component in components:
                    component.dc_pred = 0
            mcu_count += 1
            for component in components:
                for block_y in range(component.v):
                    for block_x in range(component.h):
                        coeffs = _decode_block(reader, component)
                        row = mcu_y * component.v + block_y
                        col = mcu_x * component.h + block_x
                        if row < component.blocks.shape[0] and col < component.blocks.shape[1]:
                            component.blocks[row, col] = coeffs


def _decode_block(reader: _BitReader, component: _Component) -> np.ndarray:
    coefficients = np.zeros(64, dtype=np.float32)

    category = component.dc_table.decode(reader)
    if category:
        difference = _extend(reader.bits(category), category)
        component.dc_pred += difference
    coefficients[0] = component.dc_pred

    index = 1
    while index < 64:
        run_length = component.ac_table.decode(reader)
        if run_length == 0:
            break
        size = run_length & 0x0F
        run = run_length >> 4
        if size == 0:
            if run == 15:
                index += 16
                continue
            break
        index += run
        if index >= 64:
            break
        coefficients[index] = _extend(reader.bits(size), size)
        index += 1

    quant = component.quant
    return coefficients


def _assemble(components: List[_Component], width: int, height: int,
              adobe_transform: Optional[int], component_count: int) -> np.ndarray:
    if not components or width == 0 or height == 0:
        raise JpegError("JPEG is missing frame dimensions")

    max_h = max(c.h for c in components)
    max_v = max(c.v for c in components)
    planes = []
    for component in components:
        quant_table = getattr(component, "quant_table", None)
        if quant_table is None:
            raise JpegError("internal error: quantisation table not attached")
        dequantised = component.blocks * quant_table.reshape(1, 1, 64)
        coeffs = dequantised.reshape(-1, 8, 8)
        spatial = np.einsum("nuv,uvxy->nxy", coeffs, _IDCT_BASIS, optimize=True)
        spatial = np.clip(spatial + 128.0, 0, 255)

        strips = []
        for block_row in range(component.block_h):
            start = block_row * component.block_w
            strip = spatial[start:start + component.block_w]
            strips.append(np.concatenate([strip[i] for i in range(strip.shape[0])],
                                         axis=1))
        plane = (np.concatenate(strips, axis=0) if strips
                 else np.zeros((0, 0), dtype=np.float32))

        if component.h != max_h or component.v != max_v:
            plane = _upsample(plane, max(1, max_h // component.h),
                              max(1, max_v // component.v))
        planes.append(plane[:height, :width])

    if len(planes) == 1:
        grey = planes[0].astype(np.uint8)
        return np.repeat(grey[:, :, None], 3, axis=2)

    y = planes[0]
    if len(planes) >= 3:
        cb, cr = planes[1], planes[2]
        if adobe_transform == 0:
            # RGB-encoded JPEG stored in YCbCr slots.
            rgb = np.stack([y, cb, cr], axis=2)
            return np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        cb = cr = np.zeros_like(y)

    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    rgb = np.stack([r, g, b], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _upsample(plane: np.ndarray, factor_x: int, factor_y: int) -> np.ndarray:
    """Nearest-neighbour chroma upsampling (fast and adequate for preview)."""
    if factor_x > 1:
        plane = np.repeat(plane, factor_x, axis=1)
    if factor_y > 1:
        plane = np.repeat(plane, factor_y, axis=0)
    return plane
