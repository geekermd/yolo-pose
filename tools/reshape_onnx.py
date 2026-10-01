#!/usr/bin/env python3
"""Re-shape a YOLOv8-pose ONNX model to a different square input size.

The exported graph is almost resolution-agnostic: the backbone convolutions are
shape-free, every head ``Reshape`` uses ``-1`` for the flattened spatial axis, the
upsampling ``Resize`` nodes take their target from a dynamic ``Shape``, and the
``Split`` constants are channel counts.  Only the two DFL reshapes carry a literal
anchor count, which is ``sum((S/stride)**2)`` for the three detection strides
(8/16/32) at input size ``S``.

Rewriting those two tensors is therefore enough to produce a working model at a
smaller input, with no PyTorch round-trip.

Usage::

    python tools/reshape_onnx.py models/yolov8n-pose.onnx -s 320 -o models/yolov8n-pose-320.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from yolopose.core.onnx_graph import load_model  # noqa: E402

STRIDES = (8, 16, 32)


def anchor_count(size: int) -> int:
    """Number of anchors the pose head emits for a square ``size`` input."""
    total = 0
    for stride in STRIDES:
        grid = size // stride
        total += grid * grid
    return total


def find_size_dependent_tensors(model, size: int) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Locate initializers that mention the *original* anchor count.

    Returns ``(name, old, new)`` triples.  Detection works from the graph's own
    declared input size rather than from an assumed 640.
    """
    declared = None
    for value in model.inputs:
        if value.shape and len(value.shape) == 4 and all(
                isinstance(d, int) and d > 0 for d in value.shape[2:]):
            if value.shape[2] == value.shape[3]:
                declared = int(value.shape[2])
                break
    if declared is None:
        raise SystemExit("could not determine the model's input size")

    old_anchors = anchor_count(declared)
    new_anchors = anchor_count(size)
    if old_anchors == new_anchors:
        return []

    changes: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for name, array in model.initializers.items():
        if array.dtype.kind not in "iu" or array.size == 0 or array.size > 16:
            continue
        if array.size == 1 and int(array.reshape(-1)[0]) == new_anchors:
            # Already correct (e.g. a partially converted file).
            continue
        if not (array == old_anchors).any():
            continue
        updated = array.copy()
        updated[updated == old_anchors] = new_anchors
        changes.append((name, array, updated))

    print(f"  input size      : {declared} -> {size}")
    print(f"  anchors         : {old_anchors} -> {new_anchors}")
    for name, old, new in changes:
        print(f"  rewrite {name[:44]:44} {old.reshape(-1).tolist()} -> "
              f"{new.reshape(-1).tolist()}")
    return changes


def _flatten(parts: np.ndarray) -> np.ndarray:
    return parts.reshape(-1)


def _matches(array: np.ndarray, reference: np.ndarray, old_anchors: int) -> bool:
    """True when ``array`` equals ``reference`` broadcast/reshaped to its shape."""
    try:
        candidate = reference.reshape(array.shape)
    except ValueError:
        return False
    return array.shape[-1] == old_anchors and np.array_equal(array, candidate)


def _segments(size: int, maker):
    return [maker(size // stride, stride) for stride in STRIDES]


def grid_index(size: int, dtype=np.float32) -> np.ndarray:
    """Raw ``(2, N)`` grid indices: (0..n-1) row-major, concatenated by stride."""
    parts = []
    for stride in STRIDES:
        grid = size // stride
        ys, xs = np.mgrid[0:grid, 0:grid]
        parts.append(np.stack([xs.ravel().astype(dtype), ys.ravel().astype(dtype)],
                              axis=0))
    return np.concatenate(parts, axis=1)


def grid_scaled(size: int, dtype=np.float32) -> np.ndarray:
    """``(2, N)`` grid indices multiplied by their stride."""
    parts = []
    for stride in STRIDES:
        grid = size // stride
        ys, xs = np.mgrid[0:grid, 0:grid]
        parts.append(np.stack([(xs.ravel() * stride).astype(dtype),
                               (ys.ravel() * stride).astype(dtype)], axis=0))
    return np.concatenate(parts, axis=1)


def centres(size: int, dtype=np.float32) -> np.ndarray:
    """``(2, N)`` grid centres, i.e. index + 0.5 on both axes."""
    parts = []
    for stride in STRIDES:
        grid = size // stride
        ys, xs = np.mgrid[0:grid, 0:grid]
        parts.append(np.stack([(xs.ravel() + 0.5).astype(dtype),
                               (ys.ravel() + 0.5).astype(dtype)], axis=0))
    return np.concatenate(parts, axis=1)


def strides_only(size: int, dtype=np.float32) -> np.ndarray:
    """``(N,)`` stride value repeated once per anchor."""
    return np.concatenate([
        np.full((size // stride) ** 2, float(stride), dtype=dtype)
        for stride in STRIDES])


def anchor_grid(size: int, dtype=np.float32) -> np.ndarray:
    """Rebuild the ``(2, anchors)`` anchor-coordinate tensor for ``size``.

    The exporter bakes this tensor into the head.  Its values are the *raw grid
    indices* (not multiplied by the stride) for each detection level, laid out
    row-major and concatenated in stride order 8, 16, 32.  Verified to reproduce
    the shipped 640x640 tensor exactly.
    """
    parts = []
    for stride in STRIDES:
        grid = size // stride
        ys, xs = np.mgrid[0:grid, 0:grid]
        parts.append(np.stack([xs.ravel().astype(dtype),
                               ys.ravel().astype(dtype)], axis=0))
    return np.concatenate(parts, axis=1)


def rewrite(model_bytes: bytes, size: int) -> bytes:
    """Patch the raw ONNX bytes in place.

    The initializers are rewritten byte-for-byte: the raw_data payload is located
    and the anchor values replaced, which avoids needing to re-serialise the
    protobuf.
    """
    import struct

    import onnx  # imported lazily so the tool degrades gracefully
    from onnx import numpy_helper

    model = onnx.load_from_string(model_bytes)
    declared = None
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4 and dims[2].dim_value == dims[3].dim_value:
            declared = dims[2].dim_value
            break
    if declared is None:
        raise SystemExit("could not read the model input size")

    old_input = int(declared)
    old_anchors = anchor_count(old_input)
    new_anchors = anchor_count(size)

    patched = 0
    grid_patched = 0
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 4:
            dims[2].dim_value = size
            dims[3].dim_value = size

    # The exporter emits these as *Constant nodes* rather than graph
    # initializers, so both places are walked.  Every anchor-dependent tensor is
    # rebuilt from the rules below, all of which reproduce the shipped 640x640
    # tensors bit-for-bit (verified with np.array_equal).
    def rebuild(array, name):
        """Return the resized tensor, or ``None`` when it is not anchor-related."""
        if array.ndim == 0:
            return None
        flat_size = int(np.prod(array.shape))
        if flat_size == 0:
            return None
        if array.ndim >= 2 and array.shape[-1] == old_anchors:
            # Target shape: same leading axes, new anchor count.
            target = tuple(array.shape[:-1]) + (new_anchors,)

            # (1) raw grid indices, (2, N) or (1, 2, N)
            candidate = grid_index(size)
            if _matches(array, grid_index(old_input), old_anchors) \
                    and candidate.size == int(np.prod(target)):
                return candidate.reshape(target).copy()
            # (2) grid indices multiplied by the stride
            candidate = grid_scaled(size)
            if _matches(array, grid_scaled(old_input), old_anchors) \
                    and candidate.size == int(np.prod(target)):
                return candidate.reshape(target).copy()
            # (3) grid centres (index + 0.5 on both axes)
            candidate = centres(size)
            if _matches(array, centres(old_input), old_anchors) \
                    and candidate.size == int(np.prod(target)):
                return candidate.reshape(target).copy()
            # (4) per-anchor stride, broadcast over the leading axes
            if _matches(array, strides_only(old_input), old_anchors):
                values = strides_only(size)
                return np.broadcast_to(values, target).copy()
        # (5) anchor counts appearing as literals
        if array.dtype.kind in "iu" and 0 < flat_size <= 16:
            if (array.reshape(-1) == old_anchors).any():
                updated = array.copy()
                updated[updated == old_anchors] = new_anchors
                return updated
        return None

    for tensor in model.graph.initializer:
        rebuilt = rebuild(numpy_helper.to_array(tensor), tensor.name)
        if rebuilt is None:
            continue
        tensor.CopyFrom(numpy_helper.from_array(rebuilt, tensor.name))
        if rebuilt.ndim >= 2 and rebuilt.shape[-1] == new_anchors:
            grid_patched += 1
        else:
            patched += 1

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name != "value":
                continue
            rebuilt = rebuild(numpy_helper.to_array(attribute.t), attribute.t.name)
            if rebuilt is None:
                continue
            attribute.t.CopyFrom(numpy_helper.from_array(
                rebuilt, attribute.t.name))
            if rebuilt.ndim >= 2 and rebuilt.shape[-1] == new_anchors:
                grid_patched += 1
            else:
                patched += 1

    print(f"  input size : {old_input} -> {size}")
    print(f"  anchors    : {old_anchors} -> {new_anchors}")
    print(f"  tensors    : {patched} anchor-count patched, "
          f"{grid_patched} anchor tensors rebuilt")
    return model.SerializeToString()


def verify(path: str, size: int, runs: int = 6) -> Tuple[float, str]:
    """Load the model, run it, and report inference latency."""
    import time

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    tensor = np.random.RandomState(0).rand(1, 3, size, size).astype(np.float32)
    for _ in range(3):
        session.run(None, {name: tensor})
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(runs):
            session.run(None, {name: tensor})
        timings.append((time.perf_counter() - started) / runs * 1000)
    best = min(timings)
    output = session.run(None, {name: tensor})[0]
    return best, str(output.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="source .onnx model")
    parser.add_argument("-s", "--size", type=int, required=True,
                        help="target square input size, e.g. 320")
    parser.add_argument("-o", "--output", required=True, help="destination .onnx")
    parser.add_argument("--verify", action="store_true",
                        help="run the converted model and report its latency")
    args = parser.parse_args()

    if args.size % 32:
        print("warning: sizes that are not a multiple of 32 may not divide the "
              "feature strides evenly", file=sys.stderr)

    with open(args.model, "rb") as handle:
        raw = handle.read()

    print(f"converting {args.model} -> {args.output}")
    converted = rewrite(raw, args.size)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as handle:
        handle.write(converted)
    print(f"  written    : {args.output} "
          f"({os.path.getsize(args.output) / 1e6:.1f} MB)")

    # Confirm our own parser can still read it.
    model = load_model(args.output)
    from yolopose.core.runtime import SUPPORTED_OPS
    missing = sorted(set(model.op_histogram()) - SUPPORTED_OPS)
    print(f"  parser     : {len(model.nodes)} nodes, "
          f"{'all operators supported' if not missing else 'MISSING ' + str(missing)}")
    print(f"  input      : {model.inputs[0].name} {model.inputs[0].shape}")

    if args.verify:
        latency, shape = verify(args.output, args.size)
        print(f"  latency    : {latency:.2f} ms ({1000 / latency:.1f} FPS)  output {shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
