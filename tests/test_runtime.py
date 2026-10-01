"""Runtime kernel tests with hand-computed expectations."""

import numpy as np
import pytest

from yolopose.core.onnx_graph import Node
from yolopose.core.runtime import (NumpyEngine, OnnxRuntimeError, OP_TABLE,
                                   _gather_spatial, _im2col, op_affine_grid,
                                   op_grid_sample, op_slice)


def node(op, inputs, outputs=("y",), **attrs):
    return Node(op, f"{op}_test", list(inputs), list(outputs), attrs)


def test_im2col_matches_naive_loop():
    rng = np.random.RandomState(0)
    x = rng.rand(1, 2, 5, 5).astype(np.float32)
    columns = _im2col(x, (3, 3), (2, 2), (1, 1), (1, 1, 1, 1))
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    expected = np.zeros((1, 18, 9), dtype=np.float32)
    index = 0
    for row in range(3):
        for col in range(3):
            patch = padded[:, :, row * 2:row * 2 + 3, col * 2:col * 2 + 3]
            expected[:, :, index] = patch.reshape(1, -1)
            index += 1
    assert np.allclose(columns, expected, atol=1e-6)


def test_conv_matches_naive_loop():
    rng = np.random.RandomState(1)
    x = rng.rand(1, 3, 8, 8).astype(np.float32)
    w = rng.rand(4, 3, 3, 3).astype(np.float32)
    b = rng.rand(4).astype(np.float32)
    out = OP_TABLE["Conv"](node("Conv", ["x", "w", "b"], pads=[1, 1, 1, 1]),
                           {"x": x, "w": w, "b": b})

    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    expected = np.zeros((1, 4, 8, 8), dtype=np.float32)
    for oy in range(8):
        for ox in range(8):
            patch = padded[:, :, oy:oy + 3, ox:ox + 3]
            expected[:, :, oy, ox] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    expected += b.reshape(1, -1, 1, 1)
    assert np.allclose(out, expected, atol=1e-4)


def test_depthwise_conv():
    rng = np.random.RandomState(2)
    x = rng.rand(1, 4, 6, 6).astype(np.float32)
    w = rng.rand(4, 1, 3, 3).astype(np.float32)
    out = OP_TABLE["Conv"](node("Conv", ["x", "w"], group=4, pads=[1, 1, 1, 1]),
                           {"x": x, "w": w})
    assert out.shape == (1, 4, 6, 6)
    reference = OP_TABLE["Conv"](node("Conv", ["x", "w"], pads=[1, 1, 1, 1]),
                                 {"x": x[:, :1], "w": w[:1]})
    assert np.allclose(out[:, :1], reference, atol=1e-4)


def test_grouped_conv_channel_split():
    rng = np.random.RandomState(3)
    x = rng.rand(1, 4, 4, 4).astype(np.float32)
    w = rng.rand(4, 2, 3, 3).astype(np.float32)
    out = OP_TABLE["Conv"](node("Conv", ["x", "w"], group=2, pads=[1, 1, 1, 1]),
                           {"x": x, "w": w})
    assert out.shape == (1, 4, 4, 4)


def test_maxpool_window_axes():
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    out = OP_TABLE["MaxPool"](node("MaxPool", ["x"], kernel_shape=[2, 2],
                                   strides=[2, 2]), {"x": x})
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out[0, 0], [[5, 7], [13, 15]])


def test_slice_negative_and_default_axis():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    # axes from an input, starts/ends from inputs
    out = op_slice(node("Slice", ["x", "s", "e", "a"]),
                   {"x": x, "s": np.array([0]), "e": np.array([2]),
                    "a": np.array([1])})
    assert out.shape == (2, 2, 4)
    assert np.allclose(out, x[:, :2])

    # negative index selects from the end: -2 on a size-4 axis is index 2
    out = op_slice(node("Slice", ["x", "s", "e", "a"]),
                   {"x": x, "s": np.array([-2]), "e": np.array([3]),
                    "a": np.array([2])})
    assert out.shape == (2, 3, 1)
    assert np.allclose(out, x[:, :, 2:3])


def test_slice_negative_step_reverses():
    x = np.arange(10, dtype=np.float32)
    out = op_slice(node("Slice", ["x", "s", "e", "a", "st"]),
                   {"x": x, "s": np.array([9]), "e": np.array([2]),
                    "a": np.array([0]), "st": np.array([-1])})
    assert np.allclose(out, x[9:2:-1])

    # ONNX clamps a negative-step `end` into [-1, dim-1] and -1 means "before
    # the first element", so it selects nothing (unlike Python's x[3::-1]).
    out = op_slice(node("Slice", ["x", "s", "e", "a", "st"]),
                   {"x": x, "s": np.array([3]), "e": np.array([-1]),
                    "a": np.array([0]), "st": np.array([-1])})
    assert out.size == 0

    # Reversing to the start is expressed with an explicit 0 end.
    out = op_slice(node("Slice", ["x", "s", "e", "a", "st"]),
                   {"x": x, "s": np.array([3]), "e": np.array([0]),
                    "a": np.array([0]), "st": np.array([-1])})
    assert np.allclose(out, [3, 2, 1])


def test_reshape_zero_and_minus_one():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    out = OP_TABLE["Reshape"](node("Reshape", ["x", "shape"]),
                              {"x": x, "shape": np.array([0, -1])})
    assert out.shape == (2, 12)


def test_unsqueeze_axes_reference_output_rank():
    x = np.array([1, 2, 3], dtype=np.float32)
    out = OP_TABLE["Unsqueeze"](node("Unsqueeze", ["x", "axes"]),
                                {"x": x, "axes": np.array([0])})
    assert out.shape == (1, 3)
    out = OP_TABLE["Unsqueeze"](node("Unsqueeze", ["x", "axes"]),
                                {"x": x, "axes": np.array([0, 2])})
    assert out.shape == (1, 3, 1)
    out = OP_TABLE["Unsqueeze"](node("Unsqueeze", ["x", "axes"]),
                                {"x": x, "axes": np.array([-1])})
    assert out.shape == (3, 1)


def test_batch_normalization_folds_to_affine():
    rng = np.random.RandomState(4)
    x = rng.rand(1, 3, 4, 4).astype(np.float32)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    bias = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    var = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    out = OP_TABLE["BatchNormalization"](
        node("BatchNormalization", ["x", "s", "b", "m", "v"], epsilon=1e-5),
        {"x": x, "s": scale, "b": bias, "m": mean, "v": var})
    expected = ((x - mean.reshape(1, -1, 1, 1))
                / np.sqrt(var.reshape(1, -1, 1, 1) + 1e-5)
                * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1))
    assert np.allclose(out, expected, atol=1e-5)


def test_concat_rank_collapse_matches_onnxruntime():
    """SCRFD concatenates (1,2) with (1,) operands; ONNX Runtime flattens them."""
    out = OP_TABLE["Concat"](node("Concat", ["a", "b", "c"], axis=0),
                             {"a": np.zeros((1, 2), np.float32),
                              "b": np.zeros((1,), np.float32),
                              "c": np.zeros((1,), np.float32)})
    assert out.shape == (4,)


def test_grid_sample_identity_is_exact():
    x = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    theta = np.zeros((1, 2, 3), dtype=np.float32)
    theta[0, 0, 0] = 1.0
    theta[0, 1, 1] = 1.0
    grid = op_affine_grid(node("AffineGrid", ["theta", "size"], align_corners=0),
                          {"theta": theta,
                           "size": np.array([1, 1, 4, 4], dtype=np.int64)})
    for mode in ("bilinear", "nearest"):
        sampled = op_grid_sample(
            node("GridSample", ["x", "grid"], mode=mode,
                 padding_mode="zeros", align_corners=0),
            {"x": x, "grid": grid})
        assert sampled.shape == x.shape
        assert np.allclose(sampled, x, atol=1e-5), mode


def test_grid_sample_zero_padding_outside():
    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    grid = np.full((1, 1, 1, 2), 5.0, dtype=np.float32)   # far outside
    sampled = op_grid_sample(
        node("GridSample", ["x", "grid"], mode="bilinear",
             padding_mode="zeros", align_corners=0), {"x": x, "grid": grid})
    assert np.allclose(sampled, 0.0)


def test_unsupported_operator_is_reported(pose_model):
    engine = NumpyEngine(pose_model)
    engine.nodes.append(node("TotallyFakeOp", ["x"]))
    with pytest.raises(OnnxRuntimeError, match="TotallyFakeOp"):
        engine._validate()


def test_silu_fusion_reduces_node_count(pose_model):
    from collections import Counter

    fused = NumpyEngine(pose_model, fuse=True)
    plain = NumpyEngine(pose_model, fuse=False)
    assert len(fused.nodes) < len(plain.nodes)
    fused_ops = Counter(n.op_type for n in fused.nodes)
    plain_ops = Counter(n.op_type for n in plain.nodes)
    assert fused_ops["Silu"] > 0
    assert fused_ops["Sigmoid"] < plain_ops["Sigmoid"]
    assert fused_ops["Mul"] < plain_ops["Mul"]
