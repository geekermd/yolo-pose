"""ONNX parser tests: field decoding, signed int64 attributes, Constant folding."""

import numpy as np
import pytest

from yolopose.core.onnx_graph import (as_signed64, iter_fields, load_model,
                                      read_varint)


def test_varint_roundtrip():
    for value in (0, 1, 127, 128, 300, 2 ** 31, 2 ** 63):
        encoded = bytearray()
        remaining = value
        while True:
            byte = remaining & 0x7F
            remaining >>= 7
            if remaining:
                encoded.append(byte | 0x80)
            else:
                encoded.append(byte)
                break
        decoded, position = read_varint(bytes(encoded), 0)
        assert decoded == value
        assert position == len(encoded)


def test_signed64_two_complement():
    assert as_signed64(0) == 0
    assert as_signed64(2 ** 64 - 1) == -1
    assert as_signed64(2 ** 64 - 2) == -2
    assert as_signed64(17) == 17


def test_iter_fields_mixed_wire_types():
    # field 1 varint=7, field 2 length-delimited b'ab'
    payload = bytes([0x08, 0x07, 0x12, 0x02]) + b"ab"
    fields = list(iter_fields(payload))
    assert fields[0][:2] == (1, 0)
    assert fields[0][2] == 7
    assert fields[1][0] == 2
    assert bytes(fields[1][2]) == b"ab"


def test_parser_matches_onnx_library(pose_model):
    """The hand-written parser must agree with the reference implementation."""
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    ours = load_model(pose_model)
    reference = onnx.load(pose_model, load_external_data=False)

    names = [i.name for i in reference.graph.initializer]
    assert names, "reference model has no initializers"
    for initializer in reference.graph.initializer:
        expected = numpy_helper.to_array(initializer)
        actual = ours.initializers[initializer.name]
        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected), initializer.name


def test_opset_and_io_shapes(pose_model):
    model = load_model(pose_model)
    assert model.default_opset >= 11
    assert model.inputs[0].name
    assert len(model.outputs) >= 1
    shape = model.inputs[0].shape
    assert shape[1] == 3
    assert model.op_histogram().get("Conv", 0) > 0
