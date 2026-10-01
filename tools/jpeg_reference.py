"""Independent reference JPEG decoder used only to validate yolopose.vision.jpeg.

This intentionally uses a different code structure (matrix-multiply IDCT via
scipy-free separable matrices, coefficient-level output) so that a bug in the
production decoder cannot be mirrored here.
"""
import math
import sys

import numpy as np

ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]

# Separable IDCT matrices: idct = A^T @ C @ A  with A[x][u] = c(u) cos((2x+1)u pi/16)/2
A = np.zeros((8, 8))
for x in range(8):
    for u in range(8):
        cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
        A[x][u] = 0.5 * cu * math.cos((2 * x + 1) * u * math.pi / 16.0)


class RefBitReader:
    def __init__(self, data, pos):
        self.data = list(data)
        self.pos = pos
        self.cur = 0
        self.nbits = 0

    def _next_byte(self):
        if self.pos >= len(self.data):
            return None
        b = self.data[self.pos]
        self.pos += 1
        if b == 0xFF:
            if self.pos < len(self.data) and self.data[self.pos] == 0x00:
                self.pos += 1
                return 0xFF
            return None
        return b

    def bit(self):
        if self.nbits == 0:
            b = self._next_byte()
            if b is None:
                return 0
            self.cur = b
            self.nbits = 8
        self.nbits -= 1
        return (self.cur >> self.nbits) & 1

    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


class RefHuffman:
    def __init__(self, counts, values):
        # Build (code, length) -> value table exactly per T.81 Annex C.
        self.table = {}
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                self.table[(length, code)] = values[k]
                k += 1
                code += 1
            code <<= 1
        self.maxlen = max((l for l in range(1, 17) if counts[l - 1]), default=0)

    def decode(self, reader):
        code = 0
        for length in range(1, self.maxlen + 1):
            code = (code << 1) | reader.bit()
            v = self.table.get((length, code))
            if v is not None:
                return v
        raise ValueError("bad huffman code")


def decode_reference(data):
    """Return a list of (component_index, block_row, block_col, coeffs[64])."""
    quant = {}
    huff = {}
    comps = []
    width = height = 0
    restart_interval = 0
    pos = 2
    scans = []
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        m = data[pos + 1]
        pos += 2
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            continue
        if m == 0xD9:
            break
        ln = int.from_bytes(data[pos:pos + 2], "big")
        seg = data[pos + 2:pos + ln]
        pos += ln
        if m == 0xDB:
            p = 0
            while p < len(seg):
                tid = seg[p]
                p += 1
                tbl = np.frombuffer(seg[p:p + 64], dtype=np.uint8).astype(np.int32)
                p += 64
                nat = np.zeros(64, dtype=np.int32)
                nat[ZIGZAG] = tbl
                quant[tid & 15] = nat
        elif m in (0xC0, 0xC1):
            height = int.from_bytes(seg[1:3], "big")
            width = int.from_bytes(seg[3:5], "big")
            n = seg[5]
            comps = []
            for i in range(n):
                b = 6 + i * 3
                comps.append({"id": seg[b], "h": seg[b + 1] >> 4, "v": seg[b + 1] & 15,
                              "q": seg[b + 2]})
        elif m == 0xC4:
            p = 0
            while p < len(seg):
                tid = seg[p]
                counts = list(seg[p + 1:p + 17])
                n = sum(counts)
                values = list(seg[p + 17:p + 17 + n])
                p += 17 + n
                huff[(tid >> 4, tid & 15)] = RefHuffman(counts, values)
        elif m == 0xDD:
            restart_interval = int.from_bytes(seg[0:2], "big")
        elif m == 0xDA:
            n = seg[0]
            for i in range(n):
                comp = next(c for c in comps if c["id"] == seg[1 + i * 2])
                comp["dc"] = huff[(0, seg[2 + i * 2] >> 4)]
                comp["ac"] = huff[(1, seg[2 + i * 2] & 15)]
            scans.append((pos, [next(c for c in comps if c["id"] == seg[1 + i * 2])
                                for i in range(n)]))
            break
    return {
        "width": width, "height": height, "comps": comps, "quant": quant,
        "scan": scans[0] if scans else None, "restart": restart_interval,
    }


def decode_coefficients(data):
    """Decode all DCT coefficient blocks, returning {(comp, r, c): coeffs[64]}."""
    meta = decode_reference(data)
    offset, scan_comps = meta["scan"]
    width, height, comps = meta["width"], meta["height"], meta["comps"]
    max_h = max(c["h"] for c in comps)
    max_v = max(c["v"] for c in comps)
    mcu_w, mcu_h = 8 * max_h, 8 * max_v
    mcus_x = (width + mcu_w - 1) // mcu_w
    mcus_y = (height + mcu_h - 1) // mcu_h

    reader = RefBitReader(data, offset)
    out = {}
    for comp_index, comp in enumerate(comps):
        comp["pred"] = 0
        comp["bw"] = mcus_x * comp["h"]
        comp["bh"] = mcus_y * comp["v"]
        out[comp_index] = {}

    mcu = 0
    for my in range(mcus_y):
        for mx in range(mcus_x):
            if meta["restart"] and mcu and mcu % meta["restart"] == 0:
                reader.nbits = 0
                for comp in comps:
                    comp["pred"] = 0
            mcu += 1
            for comp_index, comp in enumerate(comps):
                for by in range(comp["v"]):
                    for bx in range(comp["h"]):
                        coeffs = np.zeros(64, dtype=np.int32)
                        cat = comp["dc"].decode(reader)
                        diff = 0
                        if cat:
                            v = reader.bits(cat)
                            diff = v if v >= (1 << (cat - 1)) else v - (1 << cat) + 1
                        comp["pred"] += diff
                        coeffs[0] = comp["pred"]
                        k = 1
                        while k < 64:
                            rs = comp["ac"].decode(reader)
                            if rs == 0:
                                break
                            run, size = rs >> 4, rs & 15
                            if size == 0:
                                if run == 15:
                                    k += 16
                                    continue
                                break
                            k += run
                            if k >= 64:
                                break
                            v = reader.bits(size)
                            coeffs[k] = v if v >= (1 << (size - 1)) else v - (1 << size) + 1
                            k += 1
                        row = my * comp["v"] + by
                        col = mx * comp["h"] + bx
                        out[comp_index][(row, col)] = coeffs
    return meta, out


def reconstruct(meta, coefficients):
    """Full IDCT reconstruction to an RGB uint8 image (nearest chroma upsample)."""
    width, height = meta["width"], meta["height"]
    comps = meta["comps"]
    max_h = max(c["h"] for c in comps)
    max_v = max(c["v"] for c in comps)
    planes = []
    for ci, comp in enumerate(comps):
        q = meta["quant"][comp["q"]].reshape(8, 8)
        bh, bw = comp["bh"], comp["bw"]
        plane = np.zeros((bh * 8, bw * 8), dtype=np.float64)
        for (r, c), coeffs in coefficients[ci].items():
            coef = coeffs.reshape(8, 8).astype(np.float64) * q
            block = A.T @ coef @ A
            plane[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = block
        if comp["h"] != max_h or comp["v"] != max_v:
            plane = np.repeat(np.repeat(plane, max_v // comp["v"], axis=0),
                              max_h // comp["h"], axis=1)
        planes.append(plane[:height, :width] + 128.0)

    if len(planes) == 1:
        grey = np.clip(planes[0], 0, 255).astype(np.uint8)
        return np.repeat(grey[:, :, None], 3, axis=2)
    y, cb, cr = planes[0], planes[1], planes[2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    path = sys.argv[1]
    from PIL import Image
    meta, coeffs = decode_coefficients(open(path, "rb").read())
    mine = reconstruct(meta, coeffs)
    ref = np.asarray(Image.open(path).convert("RGB"))
    print("reference reconstruction vs Pillow:", mine.shape, ref.shape)
    print("  max diff:", np.abs(mine.astype(int) - ref.astype(int)).max())
    # print first luma block DC values
    print("  first 4 luma block DCs:",
          [int(coeffs[0][(0, c)][0]) for c in range(4)])
