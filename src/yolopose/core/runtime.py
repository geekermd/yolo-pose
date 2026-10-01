"""A pure-NumPy executor for the ONNX operator subset used by YOLO pose models.

No native code, no ``onnxruntime``, no ``protobuf``: every kernel is written
against NumPy so the exact same wheel-free code path runs on x86-64, aarch64,
riscv64 or any other platform NumPy supports.

Convolutions go through ``sliding_window_view`` + ``matmul`` (im2col style), so
they hit BLAS and stay reasonably fast despite the interpreter overhead.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .onnx_graph import Model, Node, load_model

__all__ = ["OnnxRuntimeError", "NumpyEngine", "SUPPORTED_OPS"]


class OnnxRuntimeError(RuntimeError):
    """Raised for unsupported operators or malformed graphs."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _attr(node: Node, name: str, default=None):
    return node.attrs.get(name, default)


def _as_int_list(value, default=None) -> List[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


def _auto_pad(node: Node, spatial: int, kernel: Sequence[int],
              strides: Sequence[int], dilations: Sequence[int]) -> List[int]:
    """Resolve the ``pads`` attribute honouring ``auto_pad``.

    Returns ``[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`` (ONNX layout).
    """
    explicit = _attr(node, "pads")
    if explicit:
        return _as_int_list(explicit)
    mode = (_attr(node, "auto_pad", "NOTSET") or "NOTSET").decode() \
        if isinstance(_attr(node, "auto_pad", "NOTSET"), (bytes, bytearray)) \
        else str(_attr(node, "auto_pad", "NOTSET"))
    if mode == "NOTSET":
        return [0] * (2 * spatial)
    effective = [d * (k - 1) + 1 for d, k in zip(dilations, kernel)]
    begins, ends = [], []
    # Input spatial size is unknown here, so use SAME_UPPER/SAME_LOWER on the
    # output-offset formula; the caller patches the end padding when needed.
    for i in range(spatial):
        total = effective[i] - 1
        if mode == "SAME_UPPER":
            begins.append(total // 2)
            ends.append(total - total // 2)
        elif mode == "SAME_LOWER":
            begins.append(total - total // 2)
            ends.append(total // 2)
        elif mode == "VALID":
            begins.append(0)
            ends.append(0)
        else:
            raise OnnxRuntimeError(f"unsupported auto_pad {mode!r}")
    return begins + ends


def _out_spatial(in_shape, kernel, strides, dilations, pads) -> List[int]:
    """Output grid size for a convolution/pooling op."""
    out = []
    spatial = len(kernel)
    for i in range(spatial):
        numerator = (in_shape[2 + i] + pads[i] + pads[spatial + i]
                     - dilations[i] * (kernel[i] - 1) - 1)
        if numerator < 0:
            raise OnnxRuntimeError(
                f"kernel {tuple(kernel)} with pads {tuple(pads)} does not fit "
                f"input spatial size {tuple(in_shape[2:])}"
            )
        out.append(numerator // strides[i] + 1)
    return out


def _im2col(x: np.ndarray, kernel: Sequence[int], strides: Sequence[int],
            dilations: Sequence[int], pads: Sequence[int]) -> np.ndarray:
    """Memory-efficient im2col returning shape ``(N, C*Kh*Kw, L)``."""
    spatial = x.ndim - 2
    if pads and any(pads):
        pad_width = [(0, 0), (0, 0)]
        for i in range(spatial):
            pad_width.append((int(pads[i]), int(pads[i + spatial])))
        x = np.pad(x, pad_width, mode="constant", constant_values=0)

    n, c = x.shape[0], x.shape[1]
    out_spatial = _out_spatial(x.shape, kernel, strides, dilations, pads)

    if all(d == 1 for d in dilations):
        # sliding_window_view appends one window axis per spatial axis, so for a
        # 4-D NCHW input the result layout is
        #   (N, C, out_h, out_w, kh, kw)
        # and the flattened kernel axes are *interleaved* with the output axes.
        # Move the window axes to the end before reshaping.
        view = np.lib.stride_tricks.sliding_window_view(
            x, tuple(kernel), axis=tuple(range(2, 2 + spatial))
        )
        index = [slice(None)] * view.ndim
        for i in range(spatial):
            index[2 + i] = slice(None, None, strides[i])
            index[2 + spatial + i] = slice(0, out_spatial[i])
        view = view[tuple(index)]
        perm = [0, 1] + list(range(2 + spatial, 2 + 2 * spatial)) + list(range(2, 2 + spatial))
        view = np.transpose(view, perm)
        return np.ascontiguousarray(view).reshape(n, c * int(np.prod(kernel)), -1)

    # Dilated path: fancy-index the receptive field.
    grids = np.ix_(*[np.arange(o) * strides[i] + np.arange(kernel[i]) * dilations[i]
                     for i, o in enumerate(out_spatial)])
    patches = x[(slice(None), slice(None)) + tuple(grids)]
    # patches: (N, C, *out_spatial, *kernel)
    return patches.reshape(n, c * int(np.prod(kernel)), -1)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #

def op_conv(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    w = env[node.inputs[1]]
    b = env[node.inputs[2]] if len(node.inputs) > 2 and node.inputs[2] else None

    kernel = _as_int_list(_attr(node, "kernel_shape"), w.shape[2:])
    strides = _as_int_list(_attr(node, "strides"), [1] * len(kernel))
    dilations = _as_int_list(_attr(node, "dilations"), [1] * len(kernel))
    group = int(_attr(node, "group", 1) or 1)
    pads = _auto_pad(node, len(kernel), kernel, strides, dilations)

    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)

    if group == 1 and x.shape[1] == w.shape[1]:
        columns = _im2col(x, kernel, strides, dilations, pads)      # (N, C*K, L)
        weights = w.reshape(w.shape[0], -1)                         # (M, C*K)
        out = np.matmul(weights, columns)                           # (N, M, L)
        if b is not None:
            out += np.asarray(b, dtype=np.float32).reshape(1, -1, 1)
        grid = _out_spatial(x.shape, kernel, strides, dilations, pads)
        return out.reshape(x.shape[0], w.shape[0], *grid)

    # Grouped / depthwise: split channels and convolve each group independently.
    grid = _out_spatial(x.shape, kernel, strides, dilations, pads)
    channels_in = x.shape[1] // group
    channels_out = w.shape[0] // group
    outputs = []
    for g in range(group):
        xg = x[:, g * channels_in:(g + 1) * channels_in]
        wg = w[g * channels_out:(g + 1) * channels_out]
        columns = _im2col(xg, kernel, strides, dilations, pads)
        weights = wg.reshape(wg.shape[0], -1)
        out = np.matmul(weights, columns)
        if b is not None:
            bias = np.asarray(b, dtype=np.float32)
            out += bias[g * channels_out:(g + 1) * channels_out].reshape(1, -1, 1)
        outputs.append(out.reshape(x.shape[0], wg.shape[0], *grid))
    return np.concatenate(outputs, axis=1)


def op_conv_transpose(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """Transposed convolution (deconvolution), implemented via scatter-add."""
    x = np.asarray(env[node.inputs[0]], dtype=np.float32)
    w = np.asarray(env[node.inputs[1]], dtype=np.float32)   # (Cin, Cout, *K)
    b = env[node.inputs[2]] if len(node.inputs) > 2 and node.inputs[2] else None

    spatial = w.ndim - 2
    kernel = _as_int_list(_attr(node, "kernel_shape"), w.shape[2:])
    strides = _as_int_list(_attr(node, "strides"), [1] * spatial)
    dilations = _as_int_list(_attr(node, "dilations"), [1] * spatial)
    group = int(_attr(node, "group", 1) or 1)
    pads = _auto_pad(node, spatial, kernel, strides, dilations)
    out_pad = _as_int_list(_attr(node, "output_padding"), [0] * spatial)

    n = x.shape[0]
    eff = [dilations[i] * (kernel[i] - 1) + 1 for i in range(spatial)]
    full = [strides[i] * (x.shape[2 + i] - 1) + eff[i] + out_pad[i] for i in range(spatial)]
    out_shape = [full[i] + pads[i] + pads[spatial + i] for i in range(spatial)]
    out = np.zeros([n, w.shape[1]] + out_shape, dtype=np.float32)

    for i in range(x.shape[2]):
        for j in range(x.shape[3]):
            patch = x[:, :, i, j]                                  # (N, Cin)
            scattered = np.einsum("nc,coab->noab", patch, w)       # (N, Cout, Kh, Kw)
            oy = i * strides[0]
            ox = j * strides[1]
            out[:, :, oy:oy + eff[0]:dilations[0],
                ox:ox + eff[1]:dilations[1]] += scattered
    if pads[0] or pads[1]:
        slices = [slice(None), slice(None)]
        for i in range(spatial):
            slices.append(slice(pads[i], out_shape[i] - pads[spatial + i]))
        out = out[tuple(slices)]
    if b is not None:
        out += np.asarray(b, dtype=np.float32).reshape(1, -1, *([1] * spatial))
    return out


def op_maxpool(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(env[node.inputs[0]], dtype=np.float32)
    kernel = _as_int_list(_attr(node, "kernel_shape"))
    spatial = len(kernel)
    strides = _as_int_list(_attr(node, "strides"), [1] * spatial)
    dilations = _as_int_list(_attr(node, "dilations"), [1] * spatial)
    ceil_mode = int(_attr(node, "ceil_mode", 0) or 0)
    pads = _as_int_list(_attr(node, "pads"), [0] * (2 * spatial))

    if pads and any(pads):
        pad_width = [(0, 0), (0, 0)] + [(int(pads[i]), int(pads[i + spatial]))
                                        for i in range(spatial)]
        x = np.pad(x, pad_width, mode="constant", constant_values=-np.inf)

    out_spatial = []
    for i in range(spatial):
        numerator = x.shape[2 + i] - dilations[i] * (kernel[i] - 1) - 1
        if ceil_mode:
            out_spatial.append(int(np.ceil(numerator / strides[i])) + 1)
        else:
            out_spatial.append(numerator // strides[i] + 1)

    if all(d == 1 for d in dilations):
        view = np.lib.stride_tricks.sliding_window_view(
            x, tuple(kernel), axis=tuple(range(2, 2 + spatial)))
        index = [slice(None)] * view.ndim
        for i in range(spatial):
            index[2 + i] = slice(None, None, strides[i])
            index[2 + spatial + i] = slice(0, out_spatial[i])
        view = view[tuple(index)]
        # Window axes sit at 2+spatial .. 2+2*spatial (see _im2col).
        pooled = view.max(axis=tuple(range(2 + spatial, 2 + 2 * spatial)))
        return pooled.astype(np.float32)

    grids = np.ix_(*[np.arange(o) * strides[i] + np.arange(kernel[i]) * dilations[i]
                     for i, o in enumerate(out_spatial)])
    patches = x[(slice(None), slice(None)) + tuple(grids)]
    return patches.max(axis=tuple(range(2 + spatial, 2 + 2 * spatial))).astype(np.float32)


def op_resize(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(env[node.inputs[0]], dtype=np.float32)
    mode = str(_attr(node, "mode", "nearest"))
    coord = str(_attr(node, "coordinate_transformation_mode", "half_pixel"))
    nearest_mode = str(_attr(node, "nearest_mode", "round_prefer_floor"))

    scales: Optional[np.ndarray] = None
    sizes: Optional[np.ndarray] = None
    if len(node.inputs) > 2 and node.inputs[2]:
        scales = np.asarray(env[node.inputs[2]], dtype=np.float64).reshape(-1)
    if len(node.inputs) > 3 and node.inputs[3]:
        sizes = np.asarray(env[node.inputs[3]], dtype=np.int64).reshape(-1)

    spatial = x.ndim - 2
    out_shape = list(x.shape[:2])
    if sizes is not None:
        out_shape = [int(s) for s in sizes]
    elif scales is not None:
        out_shape = [int(np.floor(x.shape[i] * scales[i])) for i in range(x.ndim)]
    else:
        raise OnnxRuntimeError("Resize requires either scales or sizes")

    # ROI (used by roi_align / tf_crop_and_resize) is not needed for YOLO.
    axes_out = []
    for axis in range(x.ndim):
        in_size = x.shape[axis]
        out_size = out_shape[axis]
        if in_size == out_size:
            axes_out.append(np.arange(in_size, dtype=np.float64))
            continue
        scale = out_size / in_size
        idx = np.arange(out_size, dtype=np.float64)
        if coord == "half_pixel":
            src = (idx + 0.5) / scale - 0.5
        elif coord == "pytorch_half_pixel":
            src = np.where(out_size > 1, (idx + 0.5) / scale - 0.5, 0.0)
        elif coord == "align_corners":
            src = idx * (in_size - 1) / max(out_size - 1, 1)
        elif coord == "asymmetric":
            src = idx / scale
        elif coord == "tf_half_pixel_for_nn":
            src = (idx + 0.5) / scale
        else:
            src = (idx + 0.5) / scale - 0.5
        axes_out.append(src)

    if spatial == 2 and mode == "nearest":
        y = axes_out[x.ndim - 2]
        xx = axes_out[x.ndim - 1]
        if nearest_mode == "floor":
            iy = np.floor(y).astype(np.int64)
            ix = np.floor(xx).astype(np.int64)
        elif nearest_mode == "ceil":
            iy = np.ceil(y).astype(np.int64)
            ix = np.ceil(xx).astype(np.int64)
        elif nearest_mode == "round_prefer_ceil":
            iy = np.rint(y).astype(np.int64)
            ix = np.rint(xx).astype(np.int64)
        else:  # round_prefer_floor
            iy = np.floor(y + 0.5).astype(np.int64)
            ix = np.floor(xx + 0.5).astype(np.int64)
        iy = np.clip(iy, 0, x.shape[x.ndim - 2] - 1)
        ix = np.clip(ix, 0, x.shape[x.ndim - 1] - 1)
        return x[:, :, iy][:, :, :, ix].astype(np.float32)

    if spatial == 2 and mode in ("linear", "bilinear"):
        y = np.clip(axes_out[x.ndim - 2], 0, x.shape[x.ndim - 2] - 1)
        xx = np.clip(axes_out[x.ndim - 1], 0, x.shape[x.ndim - 1] - 1)
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(xx).astype(np.int64)
        y1 = np.clip(y0 + 1, 0, x.shape[x.ndim - 2] - 1)
        x1 = np.clip(x0 + 1, 0, x.shape[x.ndim - 1] - 1)
        wy = (y - y0).reshape(1, 1, -1, 1)
        wx = (xx - x0).reshape(1, 1, 1, -1)
        top = x[:, :, y0][:, :, :, x0] * (1 - wx) + x[:, :, y0][:, :, :, x1] * wx
        bottom = x[:, :, y1][:, :, :, x0] * (1 - wx) + x[:, :, y1][:, :, :, x1] * wx
        return (top * (1 - wy) + bottom * wy).astype(np.float32)

    if mode == "nearest":
        # Generic nearest for >2 spatial dims.
        index = [slice(None), slice(None)]
        for axis in range(2, x.ndim):
            src = axes_out[axis]
            idx = (np.floor(src + 0.5) if nearest_mode == "round_prefer_floor"
                   else np.rint(src)).astype(np.int64)
            index.append(np.clip(idx, 0, x.shape[axis] - 1))
        return x[tuple(index)].astype(np.float32)

    raise OnnxRuntimeError(f"Resize mode {mode!r} not implemented")


def op_softmax(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    axis = int(_attr(node, "axis", -1))
    if axis < 0:
        axis += x.ndim
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32)


def op_concat(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    axis = int(_attr(node, "axis", 1))
    arrays = [env[name] for name in node.inputs if name]
    if not arrays:
        raise OnnxRuntimeError("Concat without inputs")

    # Real-world exports (e.g. InsightFace SCRFD) concatenate operands of
    # differing rank along axis 0.  ONNX Runtime tolerates this by collapsing
    # every operand to the *lowest* rank present — (1,2) + (1,) + (1,) yields
    # (4,) — so mirror that instead of rejecting the graph.
    min_rank = min(a.ndim for a in arrays)
    if any(a.ndim != min_rank for a in arrays) and min_rank <= 1:
        arrays = [a.reshape(-1) if a.ndim != min_rank else a for a in arrays]

    if axis < 0:
        axis += arrays[0].ndim
    try:
        result = np.concatenate(arrays, axis=axis)
    except (OverflowError, ValueError) as exc:
        raise OnnxRuntimeError(
            f"Concat axis={axis} on shapes {[a.shape for a in arrays]}: {exc}"
        ) from None
    dtype = arrays[0].dtype
    return result.astype(dtype) if result.dtype != dtype else result


def op_split(node: Node, env: Dict[str, np.ndarray]):
    x = env[node.inputs[0]]
    axis = int(_attr(node, "axis", 0))
    if axis < 0:
        axis += x.ndim
    if len(node.inputs) > 1 and node.inputs[1]:
        sizes = np.asarray(env[node.inputs[1]], dtype=np.int64).reshape(-1).tolist()
    else:
        sizes = _as_int_list(_attr(node, "split"))
    if not sizes:
        count = len(node.outputs)
        if count == 0:
            raise OnnxRuntimeError("Split without outputs")
        sizes = [x.shape[axis] // count] * count
        sizes[-1] += x.shape[axis] - sum(sizes)
    return np.split(x, np.cumsum(sizes)[:-1], axis=axis)


def op_slice(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    axes = _as_int_list(_attr(node, "axes"))
    starts = _as_int_list(_attr(node, "starts"))
    ends = _as_int_list(_attr(node, "ends"))
    steps = _as_int_list(_attr(node, "steps"))

    def _fetch(index: int, inline_name: str, attr_value, converter):
        if len(node.inputs) > index and node.inputs[index]:
            return converter(np.asarray(env[node.inputs[index]]).reshape(-1))
        return attr_value

    starts = _fetch(1, "starts", starts, lambda a: [int(v) for v in a])
    ends = _fetch(2, "ends", ends, lambda a: [int(v) for v in a])
    if len(node.inputs) > 3 and node.inputs[3]:
        axes = [int(v) for v in np.asarray(env[node.inputs[3]]).reshape(-1)]
    if len(node.inputs) > 4 and node.inputs[4]:
        steps = [int(v) for v in np.asarray(env[node.inputs[4]]).reshape(-1)]

    if not steps or len(steps) != len(starts):
        steps = [1] * len(starts)
    if not axes:
        axes = [0]
    if len(axes) == 1 and len(starts) > 1:
        axes = axes * len(starts)

    slices = [slice(None)] * x.ndim
    for axis, start, end, step in zip(axes, starts, ends, steps):
        axis = axis + x.ndim if axis < 0 else axis
        dim = x.shape[axis]
        step = int(step or 1)

        if step > 0:
            # ONNX clamps into [0, dim] after resolving negative indices, which
            # may come from attributes *or* from slice inputs.
            if start < 0:
                start += dim
            if end < 0:
                end += dim
            start = max(0, min(start, dim))
            end = max(0, min(end, dim))
            if end < start:
                end = start
            slices[axis] = slice(start, end, step)
            continue

        # Negative step: ONNX clamps into [-1, dim-1]; end == -1 selects nothing
        # (unlike Python, where -1 means "before the first element").
        if start < 0:
            start += dim
        start = max(-1, min(start, dim - 1))
        if end < 0:
            end += dim
        end = max(-1, min(end, dim - 1))
        if start < 0 or end < 0 or start <= end:
            slices[axis] = slice(0, 0, 1)      # definitely empty
        else:
            slices[axis] = slice(start, end, step)

    return np.ascontiguousarray(x[tuple(slices)])


def op_gather(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    data = env[node.inputs[0]]
    indices = np.asarray(env[node.inputs[1]])
    axis = int(_attr(node, "axis", 0))
    return np.take(data, indices.astype(np.int64), axis=axis)


def op_transpose(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    perm = _attr(node, "perm")
    if perm is None:
        perm = list(reversed(range(env[node.inputs[0]].ndim)))
    return np.ascontiguousarray(np.transpose(env[node.inputs[0]], _as_int_list(perm)))


def op_pad(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    pads = _as_int_list(_attr(node, "pads"))
    if len(node.inputs) > 1 and node.inputs[1]:
        pads = [int(v) for v in np.asarray(env[node.inputs[1]]).reshape(-1)]
    mode = str(_attr(node, "mode", "constant"))
    if not pads:
        return x
    spatial_rank = x.ndim
    padding = [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
    if mode == "constant":
        value = 0
        if len(node.inputs) > 2 and node.inputs[2]:
            value = float(np.asarray(env[node.inputs[2]]).reshape(-1)[0])
        return np.pad(x, padding, mode="constant", constant_values=value)
    if mode == "edge":
        return np.pad(x, padding, mode="edge")
    if mode == "reflect":
        return np.pad(x, padding, mode="reflect")
    raise OnnxRuntimeError(f"Pad mode {mode!r} not implemented")


def op_reduce_mean(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    axes = _as_int_list(_attr(node, "axes"))
    keepdims = int(_attr(node, "keepdims", 1) or 0)
    if len(node.inputs) > 1 and node.inputs[1]:
        axes = [int(v) for v in np.asarray(env[node.inputs[1]]).reshape(-1)]
    axes = [a + x.ndim if a < 0 else a for a in axes] or None
    return x.mean(axis=tuple(axes) if axes else None, keepdims=bool(keepdims)).astype(x.dtype)


def op_clip(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    lo = _attr(node, "min", None)
    hi = _attr(node, "max", None)
    if len(node.inputs) > 1 and node.inputs[1]:
        lo = float(np.asarray(env[node.inputs[1]]).reshape(-1)[0])
    if len(node.inputs) > 2 and node.inputs[2]:
        hi = float(np.asarray(env[node.inputs[2]]).reshape(-1)[0])
    return np.clip(x, lo, hi).astype(x.dtype)


# Simple elementwise kernels sharing one implementation.
def _binary(fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    def kernel(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
        a = env[node.inputs[0]]
        b = env[node.inputs[1]]
        return fn(a, b)
    return kernel


def _unary(fn: Callable[[np.ndarray], np.ndarray], dtype=np.float32):
    def kernel(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
        return fn(env[node.inputs[0]]).astype(dtype)
    return kernel


def op_shape(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    start = int(_attr(node, "start", 0) or 0)
    end = _attr(node, "end", None)
    end = x.ndim if end is None else int(end)
    if start < 0:
        start += x.ndim
    if end < 0:
        end += x.ndim
    return np.asarray(x.shape[start:end], dtype=np.int64)


def op_constant_of_shape(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    shape = [int(v) for v in np.asarray(env[node.inputs[0]]).reshape(-1)]
    value = _attr(node, "value", None)
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return np.full(shape, value.reshape(-1)[0]).astype(value.dtype)


def op_expand(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    shape = [int(v) for v in np.asarray(env[node.inputs[1]]).reshape(-1)]
    return (x * np.ones(shape, dtype=x.dtype)).astype(x.dtype)


def op_where(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    cond = env[node.inputs[0]]
    return np.where(cond, env[node.inputs[1]], env[node.inputs[2]])


def op_matmul(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    a = env[node.inputs[0]]
    b = env[node.inputs[1]]
    return np.matmul(a, b).astype(np.float32)


def op_gemm(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    a = env[node.inputs[0]]
    b = env[node.inputs[1]]
    c = env[node.inputs[2]] if len(node.inputs) > 2 and node.inputs[2] else None
    alpha = float(_attr(node, "alpha", 1.0) or 1.0)
    beta = float(_attr(node, "beta", 1.0) or 1.0)
    if int(_attr(node, "transA", 0) or 0):
        a = a.T
    if int(_attr(node, "transB", 0) or 0):
        b = b.T
    out = alpha * np.matmul(a, b)
    if c is not None:
        out = out + beta * c
    return out.astype(np.float32)


def op_flatten(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = env[node.inputs[0]]
    axis = int(_attr(node, "axis", 1))
    if axis < 0:
        axis += x.ndim
    return x.reshape(int(np.prod(x.shape[:axis], dtype=np.int64)),
                     int(np.prod(x.shape[axis:], dtype=np.int64)))


def op_reshape(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """Reshape with ONNX semantics: 0 copies the input dim, -1 is inferred."""
    x = env[node.inputs[0]]
    if len(node.inputs) > 1 and node.inputs[1]:
        shape = [int(v) for v in np.asarray(env[node.inputs[1]]).reshape(-1)]
    else:
        shape = _as_int_list(_attr(node, "shape"))
    allowzero = int(_attr(node, "allowzero", 0) or 0)

    resolved: List[int] = []
    for axis, dim in enumerate(shape):
        if dim == 0 and not allowzero:
            resolved.append(x.shape[axis] if axis < x.ndim else 1)
        else:
            resolved.append(dim)

    if -1 in resolved:
        known = 1
        for dim in resolved:
            if dim != -1:
                known *= dim
        resolved[resolved.index(-1)] = (x.size // known) if known else 1
    return np.ascontiguousarray(x.reshape(resolved))


def op_identity(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    return env[node.inputs[0]]


def _read_axes(node: Node, env: Dict[str, np.ndarray], attr_name: str) -> List[int]:
    """Read an axes list from input 1 when present, else from the attribute."""
    if len(node.inputs) > 1 and node.inputs[1]:
        return [int(v) for v in np.asarray(env[node.inputs[1]]).reshape(-1)]
    return _as_int_list(_attr(node, attr_name))


def _normalise_axes(axes: List[int], rank: int) -> List[int]:
    """Map negative axes into ``[0, rank)`` for a tensor of the given rank."""
    return [a + rank if a < 0 else a for a in axes]


def op_unsqueeze(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """Insert singleton dimensions (opset 13+: axes come from an input).

    Axes index the *output* tensor, whose rank is ``x.ndim + len(axes)``.
    """
    x = env[node.inputs[0]]
    axes = _read_axes(node, env, "axes")
    if not axes:
        axes = list(range(x.ndim))
    output_rank = x.ndim + len(axes)
    axes = _normalise_axes(axes, output_rank)
    result = x
    for axis in sorted(axes):
        result = np.expand_dims(result, axis)
    return np.ascontiguousarray(result)


def op_squeeze(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """Remove singleton dimensions; without axes, drop every size-1 axis."""
    x = env[node.inputs[0]]
    has_axes = (len(node.inputs) > 1 and node.inputs[1]) or _attr(node, "axes")
    if not has_axes:
        return np.ascontiguousarray(np.squeeze(x))
    axes = _normalise_axes(_read_axes(node, env, "axes"), x.ndim)
    for axis in sorted(axes, reverse=True):
        if x.shape[axis] == 1:
            x = np.squeeze(x, axis=axis)
    return np.ascontiguousarray(x)


def op_batch_normalization(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """Inference-mode batch normalisation, folded to a per-channel affine map."""
    x = env[node.inputs[0]]
    scale = env[node.inputs[1]]
    bias = env[node.inputs[2]]
    mean = env[node.inputs[3]]
    variance = env[node.inputs[4]]
    epsilon = float(_attr(node, "epsilon", 1e-5) or 1e-5)

    shape = (1, -1) + (1,) * (x.ndim - 2)
    multiplier = (scale / np.sqrt(variance + epsilon)).reshape(shape)
    offset = (bias - mean * scale / np.sqrt(variance + epsilon)).reshape(shape)
    return (x * multiplier + offset).astype(x.dtype)


def _rotation_matrix(angle: np.ndarray) -> np.ndarray:
    """Build ``(N, 3, 3)`` rotation matrices about the Z axis."""
    angle = angle.reshape(-1).astype(np.float32)
    zero = np.zeros_like(angle)
    one = np.ones_like(angle)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.stack([
        np.stack([cos, -sin, zero], axis=1),
        np.stack([sin, cos, zero], axis=1),
        np.stack([zero, zero, one], axis=1),
    ], axis=1)


def _affine_matrix(theta: np.ndarray, scale: np.ndarray,
                   translation: np.ndarray) -> np.ndarray:
    """Compose ``(N, 2, 3)`` affine matrices from parts."""
    theta = theta.reshape(-1)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 1:
        scale = np.stack([scale, scale], axis=1)
    translation = np.asarray(translation, dtype=np.float32).reshape(-1, 2)
    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)
    matrix = np.stack([
        np.stack([theta * scale[:, 0], -theta * 0 + np.sin(theta) * scale[:, 0],
                  translation[:, 0]], axis=1),
        np.stack([-np.sin(theta) * scale[:, 1], theta * scale[:, 1],
                  translation[:, 1]], axis=1),
    ], axis=1)
    # The expression above keeps cos/sin terms explicit; rebuild cleanly:
    cos, sin = np.cos(theta), np.sin(theta)
    row0 = np.stack([cos * scale[:, 0], -sin * scale[:, 0], translation[:, 0]], axis=1)
    row1 = np.stack([sin * scale[:, 1], cos * scale[:, 1], translation[:, 1]], axis=1)
    return np.stack([row0, row1], axis=1).astype(np.float32)


def op_affine_grid(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """ONNX AffineGrid (opset 20): theta (N,2,3) -> sampling grid (N,H,W,2).

    Grid coordinates are normalised to ``[-1, 1]``; ``align_corners`` selects
    between pixel-centre and corner alignment.
    """
    theta = np.asarray(env[node.inputs[0]], dtype=np.float32)
    if len(node.inputs) > 1 and node.inputs[1]:
        size = [int(v) for v in np.asarray(env[node.inputs[1]]).reshape(-1)]
    else:
        size = _as_int_list(_attr(node, "size"))
    align_corners = int(_attr(node, "align_corners", 0) or 0)

    if len(size) == 4:
        height, width = size[2], size[3]
    elif len(size) == 2:
        height, width = size
    else:
        raise OnnxRuntimeError(f"AffineGrid needs a 2- or 4-element size, got {size}")

    if theta.ndim == 2:
        theta = theta[None]
    n = theta.shape[0]

    if align_corners:
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32) if height > 1 else np.zeros(1, np.float32)
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32) if width > 1 else np.zeros(1, np.float32)
    else:
        ys = (np.arange(height, dtype=np.float32) * 2.0 + 1.0) / height - 1.0
        xs = (np.arange(width, dtype=np.float32) * 2.0 + 1.0) / width - 1.0

    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    base = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=2)      # (H, W, 3)
    flat = base.reshape(-1, 3).T                                          # (3, H*W)
    transformed = np.einsum("nij,jk->nik", theta, flat)                   # (N, 2, HW)
    return np.transpose(transformed, (0, 2, 1)).reshape(n, height, width, 2).astype(np.float32)


def _gather_spatial(x: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Gather ``(N, C, H, W)`` at per-pixel ``(iy, ix)`` indices of shape ``(N, *S)``.

    Returns ``(N, C, *S)``.  Implemented with flat spatial indices: each output
    pixel selects one position along the flattened ``H*W`` axis, which keeps the
    channel axis in place (plain ``x[b, :, iy, ix]`` indexing would broadcast the
    channel slice into a trailing dimension instead).
    """
    n, channels, height, width = x.shape
    spatial = iy.shape[1:]                      # drop the batch dimension
    iy_b = np.broadcast_to(iy, (n,) + spatial).astype(np.int64)
    ix_b = np.broadcast_to(ix, (n,) + spatial).astype(np.int64)

    length = int(np.prod(spatial)) if spatial else 1
    flat_index = (iy_b * width + ix_b).reshape(n, length)             # (N, L)
    # Spatial axis moved to position 1 so the gather is unambiguous; the index
    # arrays are stacked explicitly so broadcasting follows (N, L, C).
    transposed = np.transpose(x, (0, 2, 3, 1)).reshape(n, height * width, channels)
    batch_index = np.full((n, length, channels), np.arange(n)[:, None, None],
                          dtype=np.int64)
    channel_index = np.broadcast_to(np.arange(channels, dtype=np.int64),
                                    (n, length, channels))
    spatial_index = np.broadcast_to(flat_index[:, :, None], (n, length, channels))
    gathered = transposed[batch_index, spatial_index, channel_index]
    return np.transpose(gathered, (0, 2, 1)).reshape((n, channels) + spatial)


def op_grid_sample(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """ONNX GridSample (opset 16+): bilinear/nearest resampling of NCHW data."""
    x = np.asarray(env[node.inputs[0]], dtype=np.float32)
    grid = np.asarray(env[node.inputs[1]], dtype=np.float32)
    mode = str(_attr(node, "mode", "bilinear"))
    padding_mode = str(_attr(node, "padding_mode", "zeros"))
    align_corners = int(_attr(node, "align_corners", 0) or 0)

    if grid.ndim == 3:
        grid = grid[None]
    n, out_h, out_w, _ = grid.shape
    _, channels, in_h, in_w = x.shape

    gx = grid[:, :, :, 0]
    gy = grid[:, :, :, 1]
    if align_corners:
        src_x = (gx + 1.0) * (in_w - 1) / 2.0
        src_y = (gy + 1.0) * (in_h - 1) / 2.0
    else:
        src_x = ((gx + 1.0) * in_w - 1.0) / 2.0
        src_y = ((gy + 1.0) * in_h - 1.0) / 2.0

    if mode == "nearest":
        ix = np.rint(src_x).astype(np.int64)
        iy = np.rint(src_y).astype(np.int64)
        if padding_mode == "border":
            ix = np.clip(ix, 0, in_w - 1)
            iy = np.clip(iy, 0, in_h - 1)
        valid = (ix >= 0) & (ix < in_w) & (iy >= 0) & (iy < in_h)
        ix = np.clip(ix, 0, in_w - 1)
        iy = np.clip(iy, 0, in_h - 1)
        out = _gather_spatial(x, iy, ix)
        if padding_mode != "border":
            out = out * valid[:, None, :, :]
        return out.astype(np.float32)

    if padding_mode == "border":
        src_x = np.clip(src_x, 0, in_w - 1)
        src_y = np.clip(src_y, 0, in_h - 1)
        x0 = np.floor(src_x).astype(np.int64)
        y0 = np.floor(src_y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, in_w - 1)
        y1 = np.clip(y0 + 1, 0, in_h - 1)
        x0 = np.clip(x0, 0, in_w - 1)
        y0 = np.clip(y0, 0, in_h - 1)
        weight_x = (src_x - x0)[:, None]
        weight_y = (src_y - y0)[:, None]
    else:
        # ONNX `zeros` padding treats out-of-range neighbours as 0 while still
        # interpolating.  Clamp the gather coordinates so indexing stays valid,
        # then zero the fully-outside pixels below.
        x0 = np.floor(src_x).astype(np.int64)
        y0 = np.floor(src_y).astype(np.int64)
        weight_x = (src_x - x0)[:, None]
        weight_y = (src_y - y0)[:, None]
        x1, y1 = x0 + 1, y0 + 1
        valid = (x0 >= 0) & (x0 < in_w) & (y0 >= 0) & (y0 < in_h)
        x0 = np.clip(x0, 0, in_w - 1)
        x1 = np.clip(x1, 0, in_w - 1)
        y0 = np.clip(y0, 0, in_h - 1)
        y1 = np.clip(y1, 0, in_h - 1)

    top_left = _gather_spatial(x, y0, x0)
    top_right = _gather_spatial(x, y0, x1)
    bottom_left = _gather_spatial(x, y1, x0)
    bottom_right = _gather_spatial(x, y1, x1)

    top = top_left * (1 - weight_x) + top_right * weight_x
    bottom = bottom_left * (1 - weight_x) + bottom_right * weight_x
    out = top * (1 - weight_y) + bottom * weight_y
    if padding_mode != "border":
        out = out * valid[:, None, :, :]
    return out.astype(np.float32)


def op_lrn(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    """Local response normalisation (used by some face/classification nets)."""
    x = env[node.inputs[0]].astype(np.float32)
    size = int(_attr(node, "size", 5) or 5)
    alpha = float(_attr(node, "alpha", 1e-4) or 1e-4)
    beta = float(_attr(node, "beta", 0.75) or 0.75)
    bias = float(_attr(node, "bias", 1.0) or 1.0)

    half = (size - 1) // 2
    squared = x * x
    padded = np.pad(squared, [(0, 0), (half, half)] + [(0, 0)] * (x.ndim - 2))
    accumulator = np.zeros_like(squared)
    for offset in range(size):
        accumulator += padded[:, offset:offset + x.shape[1]]
    return (x / np.power(bias + alpha * accumulator, beta)).astype(np.float32)


# Op name -> kernel.  Attribute defaults follow the ONNX spec.
def _make_sigmoid(_n, env, **kw):
    return (1.0 / (1.0 + np.exp(-env[_n.inputs[0]]))).astype(np.float32)


OP_TABLE = {
    "Conv": op_conv,
    "ConvTranspose": op_conv_transpose,
    "MaxPool": op_maxpool,
    "AveragePool": lambda n, e: _avgpool(n, e),
    "GlobalAveragePool": lambda n, e: e[n.inputs[0]].mean(
        axis=tuple(range(2, e[n.inputs[0]].ndim)), keepdims=True).astype(np.float32),
    "Resize": op_resize,
    "Upsample": op_resize,
    "Softmax": op_softmax,
    "LogSoftmax": lambda n, e: np.log(op_softmax(n, e)),
    "Concat": op_concat,
    "Split": op_split,
    "Slice": op_slice,
    "Gather": op_gather,
    "Transpose": op_transpose,
    "Pad": op_pad,
    "ReduceMean": op_reduce_mean,
    "ReduceSum": lambda n, e: e[n.inputs[0]].sum(
        axis=tuple(_as_int_list(_attr(n, "axes"))) or None,
        keepdims=bool(int(_attr(n, "keepdims", 1) or 0))).astype(e[n.inputs[0]].dtype),
    "Clip": op_clip,
    "Shape": op_shape,
    "ConstantOfShape": op_constant_of_shape,
    "Expand": op_expand,
    "Where": op_where,
    "MatMul": op_matmul,
    "Gemm": op_gemm,
    "Flatten": op_flatten,
    "Identity": op_identity,
    "Reshape": op_reshape,
    "Unsqueeze": op_unsqueeze,
    "Squeeze": op_squeeze,
    "BatchNormalization": op_batch_normalization,
    "AffineGrid": op_affine_grid,
    "GridSample": op_grid_sample,
    "LRN": op_lrn,
    "Sigmoid": _make_sigmoid,
    "Silu": _unary(lambda x: x / (1.0 + np.exp(-x))),
    "HardSigmoid": _unary(lambda x: np.clip(0.2 * x + 0.5, 0, 1)),
    "HardSwish": _unary(lambda x: x * np.clip(x / 6 + 0.5, 0, 1)),
    "Erf": _unary(lambda x: np.vectorize(np.math.erf)(x)) if False else
            (lambda n, e: _erf(e[n.inputs[0]])),
    "Add": _binary(lambda a, b: (a + b).astype(np.result_type(a.dtype, b.dtype))),
    "Sub": _binary(lambda a, b: (a - b).astype(np.result_type(a.dtype, b.dtype))),
    "Mul": _binary(lambda a, b: (a * b).astype(np.result_type(a.dtype, b.dtype))),
    "Div": _binary(lambda a, b: (a / b).astype(np.float32) if a.dtype.kind == "f" else a // b),
    "Pow": _binary(lambda a, b: np.power(a, b).astype(np.float32)),
    "Sqrt": _unary(np.sqrt),
    "Exp": _unary(np.exp),
    "Log": _unary(np.log),
    "Abs": _unary(np.abs),
    "Neg": _unary(np.negative),
    "Relu": _unary(lambda x: np.maximum(x, 0)),
    "LeakyRelu": lambda n, e: np.where(
        e[n.inputs[0]] > 0, e[n.inputs[0]],
        e[n.inputs[0]] * float(_attr(n, "alpha", 0.01) or 0.01)).astype(np.float32),
    "PRelu": lambda n, e: np.where(
        e[n.inputs[0]] > 0, e[n.inputs[0]], e[n.inputs[0]] * e[n.inputs[1]]).astype(np.float32),
    "Tanh": _unary(np.tanh),
    "Floor": _unary(np.floor, None),
    "Ceil": _unary(np.ceil, None),
    "Round": _unary(np.rint, None),
    "Cast": lambda n, e: e[n.inputs[0]].astype(
        np.dtype(np.int64) if int(_attr(n, "to", 1)) == 7 else np.float32),
    "Equal": _binary(lambda a, b: a == b),
    "Greater": _binary(lambda a, b: a > b),
    "Less": _binary(lambda a, b: a < b),
    "Not": _unary(np.logical_not, None),
    "And": _binary(np.logical_and),
    "Or": _binary(np.logical_or),
    "Min": _binary(np.minimum),
    "Max": _binary(np.maximum),
    "Sum": lambda n, e: sum(e[name] for name in n.inputs if name),
    "GatherElements": lambda n, e: np.take_along_axis(
        e[n.inputs[0]], np.asarray(e[n.inputs[1]]).astype(np.int64),
        axis=int(_attr(n, "axis", 0))),
    "Range": lambda n, e: np.arange(
        float(np.asarray(e[n.inputs[0]]).reshape(-1)[0]),
        float(np.asarray(e[n.inputs[1]]).reshape(-1)[0]),
        float(np.asarray(e[n.inputs[2]]).reshape(-1)[0])).astype(np.float32),
}

SUPPORTED_OPS = frozenset(OP_TABLE)


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised error function (Abramowitz & Stegun 7.1.26 is too coarse for
    SiLU-adjacent kernels, so use the tanh-based rational approximation)."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                 + t * (-1.453152027 + t * 1.061405429))))
    y = 1.0 - poly * np.exp(-ax * ax)
    return (sign * y).astype(np.float32)


def _avgpool(node: Node, env: Dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(env[node.inputs[0]], dtype=np.float32)
    kernel = _as_int_list(_attr(node, "kernel_shape"))
    spatial = len(kernel)
    strides = _as_int_list(_attr(node, "strides"), [1] * spatial)
    pads = _as_int_list(_attr(node, "pads"), [0] * (2 * spatial))
    ceil_mode = int(_attr(node, "ceil_mode", 0) or 0)
    count_include_pad = int(_attr(node, "count_include_pad", 0) or 0)

    if pads and any(pads):
        pad_width = [(0, 0), (0, 0)] + [(int(pads[i]), int(pads[i + spatial]))
                                        for i in range(spatial)]
        x = np.pad(x, pad_width, mode="constant", constant_values=0.0)
    out_spatial = []
    for i in range(spatial):
        numerator = x.shape[2 + i] - kernel[i]
        out_spatial.append((int(np.ceil(numerator / strides[i])) + 1) if ceil_mode
                           else (numerator // strides[i] + 1))
    view = np.lib.stride_tricks.sliding_window_view(
        x, tuple(kernel), axis=tuple(range(2, 2 + spatial)))
    index = [slice(None)] * view.ndim
    for i in range(spatial):
        index[2 + i] = slice(None, None, strides[i])
        index[2 + spatial + i] = slice(0, out_spatial[i])
    windowed = view[tuple(index)]
    total = windowed.sum(axis=tuple(range(2 + spatial, 2 + 2 * spatial)))
    if count_include_pad or not any(pads):
        divisor = int(np.prod(kernel))
    else:
        divisor = int(np.prod(kernel))
    return (total / divisor).astype(np.float32)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #

class NumpyEngine:
    """Executes an ONNX graph with NumPy kernels.

    Parameters
    ----------
    model:
        Path to an ``.onnx`` file, raw model bytes, or an already-loaded
        :class:`~yolopose.core.onnx_graph.Model`.
    fuse:
        Fuse ``Sigmoid``+``Mul`` pairs into a single SiLU kernel.  This removes
        roughly half of the elementwise launches in YOLOv8 backbones.
    """

    def __init__(self, model, fuse: bool = True, verbose: bool = False):
        if isinstance(model, Model):
            self.model = model
        else:
            self.model = load_model(model)
        self.verbose = verbose
        self.fuse = fuse
        self.nodes = list(self.model.nodes)
        self.weights = dict(self.model.initializers)
        if fuse:
            self.nodes = self._fuse_silu(self.nodes)
        self._validate()
        self.profile: Dict[str, float] = {}
        self.last_timing: Dict[str, float] = {}

    # -- graph preparation -------------------------------------------------- #
    @staticmethod
    def _fuse_silu(nodes: List[Node]) -> List[Node]:
        """Rewrite ``Sigmoid(x) -> Mul(x, s)`` into a single ``Silu`` kernel.

        Two passes are needed: first decide which Sigmoid nodes are fully
        absorbed (their only consumer is the matching Mul), then rebuild the
        node list without those Sigmoids.  Dropping them matters — leaving them
        in the graph would evaluate the same sigmoid twice.
        """
        consumers: Dict[str, List[Node]] = {}
        for node in nodes:
            for name in node.inputs:
                consumers.setdefault(name, []).append(node)

        sigmoid_of: Dict[str, Node] = {}
        for node in nodes:
            if node.op_type == "Sigmoid":
                sigmoid_of[node.outputs[0]] = node

        absorbed: set = set()
        replacements: Dict[int, Node] = {}
        for node in nodes:
            if node.op_type != "Mul" or len(node.inputs) < 2:
                continue
            first, second = node.inputs[0], node.inputs[1]
            sigmoid = sigmoid_of.get(first) or sigmoid_of.get(second)
            if sigmoid is None:
                continue
            other = second if sigmoid_of.get(first) else first
            if other != sigmoid.inputs[0]:
                continue
            if len(consumers.get(sigmoid.outputs[0], [])) != 1:
                continue
            absorbed.add(id(sigmoid))
            replacements[id(node)] = Node("Silu", node.name + "_silu", [other],
                                          node.outputs, {})

        return [replacements.get(id(node), node) for node in nodes
                if id(node) not in absorbed]

    def _validate(self) -> None:
        missing = {}
        for node in self.nodes:
            kernel = OP_TABLE.get(node.op_type)
            if kernel is None:
                missing[node.op_type] = missing.get(node.op_type, 0) + 1
        if missing:
            raise OnnxRuntimeError(
                "unsupported operator(s) in graph: "
                + ", ".join(f"{k} x{v}" for k, v in sorted(missing.items()))
            )

    # -- execution ---------------------------------------------------------- #
    def run(self, feeds: Dict[str, np.ndarray],
            profile: bool = False) -> Dict[str, np.ndarray]:
        """Run the graph once with the given input tensors."""
        env: Dict[str, np.ndarray] = dict(self.weights)
        for name, array in feeds.items():
            env[name] = np.ascontiguousarray(array, dtype=np.float32)

        timings: Dict[str, float] = {}
        for node in self.nodes:
            kernel = OP_TABLE[node.op_type]
            if profile:
                start = time.perf_counter()
            try:
                result = kernel(node, env)
            except Exception as exc:  # pragma: no cover - diagnostic path
                raise OnnxRuntimeError(
                    f"operator {node.op_type} ({node.name}) failed: {exc}"
                ) from exc
            if isinstance(result, (list, tuple)):
                for out_name, value in zip(node.outputs, result):
                    env[out_name] = value
            else:
                env[node.outputs[0]] = result
            if profile:
                elapsed = time.perf_counter() - start
                timings[node.op_type] = timings.get(node.op_type, 0.0) + elapsed

        self.last_timing = timings
        if profile:
            for op, seconds in timings.items():
                self.profile[op] = self.profile.get(op, 0.0) + seconds

        return {value.name: env[value.name] for value in self.model.outputs
                if value.name in env}

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        name = self.model.inputs[0].name if self.model.inputs else "images"
        return self.run({name: image})

    # -- misc --------------------------------------------------------------- #
    @property
    def input_name(self) -> str:
        return self.model.inputs[0].name if self.model.inputs else "images"

    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        if not self.model.inputs:
            return None
        shape = self.model.inputs[0].shape
        if shape is None or any(not isinstance(d, int) for d in shape):
            return None
        return tuple(int(d) for d in shape)

    def describe(self) -> str:
        from .onnx_graph import describe
        return describe(self.model)
