"""Minimal, dependency-free reader for the ONNX serialisation format.

This module decodes the protobuf wire format used by ``.onnx`` files into plain
Python/NumPy objects.  It intentionally implements only the subset of the
``ModelProto`` schema needed to execute inference graphs, which keeps the whole
runtime free of native dependencies (no ``protobuf``, no ``onnx``).

Supported: IR versions 3-9, opset 1-21, embedded and external tensor data,
graph-level constant folding of ``Constant`` nodes.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# protobuf wire format primitives
# --------------------------------------------------------------------------- #

WIRE_VARINT = 0
WIRE_I64 = 1
WIRE_LEN = 2
WIRE_I32 = 5


class OnnxFormatError(RuntimeError):
    """Raised when a byte stream cannot be decoded as ONNX."""


def read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    """Decode an unsigned LEB128 varint starting at ``pos``."""
    result = 0
    shift = 0
    n = len(buf)
    while True:
        if pos >= n:
            raise OnnxFormatError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise OnnxFormatError("varint too long")


def iter_fields(buf: bytes):
    """Iterate ``(field_number, wire_type, value)`` triples of a message.

    ``value`` is an ``int`` for varints and a ``memoryview`` slice for
    length-delimited / fixed-width fields.
    """
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = read_varint(buf, pos)
        field_no = key >> 3
        wire = key & 0x07
        if wire == WIRE_VARINT:
            value, pos = read_varint(buf, pos)
            yield field_no, wire, value
        elif wire == WIRE_I64:
            if pos + 8 > end:
                raise OnnxFormatError("truncated i64 field")
            yield field_no, wire, buf[pos:pos + 8]
            pos += 8
        elif wire == WIRE_LEN:
            length, pos = read_varint(buf, pos)
            if pos + length > end:
                raise OnnxFormatError("truncated length-delimited field")
            yield field_no, wire, buf[pos:pos + length]
            pos += length
        elif wire == WIRE_I32:
            if pos + 4 > end:
                raise OnnxFormatError("truncated i32 field")
            yield field_no, wire, buf[pos:pos + 4]
            pos += 4
        else:
            raise OnnxFormatError(f"unsupported wire type {wire}")


def collect(buf: bytes) -> Dict[int, List[Tuple[int, object]]]:
    """Group ``iter_fields`` output by field number."""
    grouped: Dict[int, List[Tuple[int, object]]] = {}
    for field_no, wire, value in iter_fields(buf):
        grouped.setdefault(field_no, []).append((wire, value))
    return grouped


def first(grouped, field_no, default=None):
    entries = grouped.get(field_no)
    if not entries:
        return default
    return entries[0][1]


def all_of(grouped, field_no) -> List[object]:
    return [value for _wire, value in grouped.get(field_no, [])]


def as_text(value) -> str:
    """Decode a length-delimited field as UTF-8 text."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", "replace")


def repeated_varints(grouped, field_no) -> List[int]:
    """Read a repeated int64 field that may be packed or unpacked."""
    out: List[int] = []
    for wire, value in grouped.get(field_no, []):
        if wire == WIRE_VARINT:
            out.append(int(value))
        elif wire == WIRE_LEN:
            pos = 0
            raw = value
            while pos < len(raw):
                item, pos = read_varint(raw, pos)
                out.append(item)
    return out


def as_signed64(value: int) -> int:
    """Reinterpret a protobuf varint as a two's-complement signed int64.

    ``int64`` fields (ONNX attributes such as ``axis = -1``) are encoded as
    10-byte varints, so a naive unsigned read yields 2**64-1 instead of -1.
    """
    value = int(value)
    if value >= 1 << 63:
        value -= 1 << 64
    return value


# --------------------------------------------------------------------------- #
# tensor data types
# --------------------------------------------------------------------------- #

ONNX_DTYPES: Dict[int, np.dtype] = {
    1: np.dtype("<f4"),    # FLOAT
    2: np.dtype("u1"),     # UINT8
    3: np.dtype("i1"),     # INT8
    4: np.dtype("<u2"),    # UINT16
    5: np.dtype("<i2"),    # INT16
    6: np.dtype("<i4"),    # INT32
    7: np.dtype("<i8"),    # INT64
    9: np.dtype("?"),      # BOOL
    10: np.dtype("<f2"),   # FLOAT16
    11: np.dtype("<f8"),   # DOUBLE
    12: np.dtype("<u4"),   # UINT32
    13: np.dtype("<u8"),   # UINT64
    16: np.dtype("|V1"),   # BFLOAT16 (kept as raw bytes, converted on demand)
}

DTYPE_NAMES: Dict[int, str] = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32",
    7: "int64", 8: "string", 9: "bool", 10: "float16", 11: "float64",
    12: "uint32", 13: "uint64", 14: "complex64", 15: "complex128",
    16: "bfloat16",
}

# Attributes whose payload is an int, float or string (field numbers in AttributeProto)
_ATTR_TYPE = {
    1: "FLOAT", 2: "INT", 3: "STRING", 4: "TENSOR", 5: "GRAPH",
    6: "FLOATS", 7: "INTS", 8: "STRINGS", 9: "TENSORS", 10: "GRAPHS",
    11: "SPARSE_TENSOR", 13: "SPARSE_TENSORS",
}


@dataclass
class Tensor:
    """A named N-dimensional array."""

    name: str
    array: np.ndarray
    dtype: Optional[np.dtype] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.array.shape)


@dataclass
class Node:
    """A single operator invocation."""

    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Node {self.op_type} {self.name} {self.inputs} -> {self.outputs}>"


@dataclass
class ValueInfo:
    name: str
    shape: Optional[List[object]]
    elem_type: int

    @property
    def type_name(self) -> str:
        return DTYPE_NAMES.get(self.elem_type, f"type{self.elem_type}")


@dataclass
class Model:
    """A decoded ONNX graph plus the metadata needed to execute it."""

    graph_name: str
    nodes: List[Node]
    initializers: Dict[str, np.ndarray]
    inputs: List[ValueInfo]
    outputs: List[ValueInfo]
    value_info: Dict[str, ValueInfo]
    opsets: Dict[str, int]
    ir_version: int
    producer: str
    model_dir: str = "."

    def op_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for node in self.nodes:
            hist[node.op_type] = hist.get(node.op_type, 0) + 1
        return hist

    @property
    def default_opset(self) -> int:
        return self.opsets.get("", max(self.opsets.values(), default=17))


# --------------------------------------------------------------------------- #
# proto -> python decoders
# --------------------------------------------------------------------------- #

def _decode_tensor(raw: bytes, model_dir: str, external_files: Dict[str, bytes]) -> np.ndarray:
    """Decode a TensorProto message into a NumPy array.

    Field numbers follow ``onnx.proto``: ``1:dims 2:data_type 3:segment
    4:float_data 5:int32_data 6:string_data 7:int64_data 8:name 9:raw_data
    10:double_data 11:uint64_data 13:external_data 14:data_location``.
    """
    grouped = collect(raw)
    dims = repeated_varints(grouped, 1)
    elem_type = int(first(grouped, 2, 1) or 1)
    name = as_text(first(grouped, 8))

    ext_data = first(grouped, 13)
    data_location = int(first(grouped, 14, 0) or 0)
    if data_location == 0 and ext_data is not None:
        loc = collect(ext_data)
        if as_text(first(loc, 1, "DEFAULT")) == "EXTERNAL":
            data_location = 1

    dtype = ONNX_DTYPES.get(elem_type)
    if dtype is None:
        raise OnnxFormatError(f"unsupported tensor element type {elem_type} for {name!r}")

    raw_bytes = first(grouped, 9)
    if raw_bytes is not None and len(raw_bytes):
        data = np.frombuffer(bytes(raw_bytes), dtype=dtype)
    elif data_location == 1:
        loc = collect(ext_data)
        rel = as_text(first(loc, 2))
        offset = int(first(loc, 3, 0) or 0)
        length = int(first(loc, 4, 0) or 0)
        blob = external_files.get(rel)
        if blob is None:
            path = os.path.join(model_dir, rel)
            if not os.path.exists(path):
                raise OnnxFormatError(
                    f"external tensor data {rel!r} referenced by {name!r} is missing; "
                    "run `yolo-pose models pull` to fetch the full model"
                )
            with open(path, "rb") as handle:
                blob = handle.read()
            external_files[rel] = blob
        if length:
            blob = blob[offset:offset + length]
        data = np.frombuffer(blob, dtype=dtype)
    else:
        floats: List[float] = []
        for wire, value in grouped.get(4, []):
            if wire == WIRE_LEN:
                floats.extend(struct.unpack(f"<{len(value) // 4}f", bytes(value)))
            else:
                floats.append(struct.unpack("<f", value)[0])
        doubles: List[float] = []
        for wire, value in grouped.get(10, []):
            if wire == WIRE_LEN:
                doubles.extend(struct.unpack(f"<{len(value) // 8}d", bytes(value)))
            else:
                doubles.append(struct.unpack("<d", value)[0])
        ints32: List[int] = []
        for wire, value in grouped.get(5, []):
            if wire == WIRE_LEN:
                ints32.extend(struct.unpack(f"<{len(value) // 4}i", bytes(value)))
            else:
                ints32.append(struct.unpack("<i", value)[0])
        # int64_data is a packed repeated varint field: apply two's complement
        # so negative constants (Slice bounds, axis indices) survive.
        ints64 = [as_signed64(v) for v in repeated_varints(grouped, 7)]
        uints64 = repeated_varints(grouped, 11)

        if floats:
            data = np.asarray(floats, dtype=dtype)
        elif doubles:
            data = np.asarray(doubles, dtype=dtype)
        elif ints64:
            data = np.asarray(ints64, dtype=dtype)
        elif uints64:
            data = np.asarray(uints64, dtype=dtype)
        elif ints32:
            data = np.asarray(ints32, dtype=dtype)
        else:
            data = np.empty([0], dtype=dtype)

    shape = [int(d) for d in dims]
    expected = 1
    for d in shape:
        expected *= d
    if shape and data.size != expected:
        raise OnnxFormatError(
            f"tensor {name!r} declares shape {shape} but carries {data.size} values"
        )
    if shape:
        data = data.reshape(shape)
    if elem_type == 16 and data.dtype == np.dtype("|V1"):
        data = data.view(np.uint16)
    return data


def _decode_attribute(raw: bytes, model_dir: str, external_files) -> Tuple[str, object]:
    grouped = collect(raw)
    name = as_text(first(grouped, 1))
    attr_type = int(first(grouped, 20, 0) or 0)
    kind = _ATTR_TYPE.get(attr_type)

    if kind is None:
        # Older files omit the type field: infer it from whichever payload exists.
        for field_no, candidate in ((2, "FLOAT"), (3, "INT"), (4, "STRING"),
                                    (5, "TENSOR"), (7, "FLOATS"), (8, "INTS"),
                                    (9, "STRINGS")):
            if field_no in grouped:
                kind = candidate
                break

    if kind == "FLOAT":
        return name, struct.unpack("<f", first(grouped, 2))[0]
    if kind == "INT":
        return name, as_signed64(int(first(grouped, 3, 0)))
    if kind == "STRING":
        return name, as_text(first(grouped, 4))
    if kind == "TENSOR":
        return name, _decode_tensor(first(grouped, 5), model_dir, external_files)
    if kind == "FLOATS":
        value = first(grouped, 7)
        return name, list(struct.unpack(f"<{len(value) // 4}f", bytes(value))) if value else []
    if kind == "INTS":
        return name, [as_signed64(v) for v in repeated_varints(grouped, 8)]
    if kind == "STRINGS":
        return name, [as_text(v) for v in all_of(grouped, 9)]
    if kind == "TENSORS":
        return name, [_decode_tensor(v, model_dir, external_files) for v in all_of(grouped, 9)]
    return name, None


def _decode_value_info(raw: bytes) -> ValueInfo:
    """Decode a ValueInfoProto: name, type (TypeProto)."""
    grouped = collect(raw)
    name = as_text(first(grouped, 1))
    type_msg = first(grouped, 2)
    elem_type = 0
    shape: Optional[List[object]] = None
    if type_msg is not None:
        tgrouped = collect(type_msg)
        tensor_type = first(tgrouped, 1)
        if tensor_type is not None:
            tt = collect(tensor_type)
            elem_type = int(first(tt, 1, 0) or 0)
            shape_msg = first(tt, 2)
            if shape_msg is not None:
                shape = []
                for dim_raw in all_of(collect(shape_msg), 1):
                    dim = collect(dim_raw)
                    if 1 in dim:                      # dim_value
                        shape.append(int(first(dim, 1)))
                    elif 2 in dim:                    # dim_param (symbolic)
                        shape.append(as_text(first(dim, 2)))
                    else:
                        shape.append(None)
    return ValueInfo(name=name, shape=shape, elem_type=elem_type)


def _decode_node(raw: bytes, model_dir: str, external_files) -> Node:
    grouped = collect(raw)
    op_type = as_text(first(grouped, 4))
    name = as_text(first(grouped, 3)) or op_type
    inputs = [as_text(v) for v in all_of(grouped, 1)]
    outputs = [as_text(v) for v in all_of(grouped, 2)]
    attrs: Dict[str, object] = {}
    for attr_raw in all_of(grouped, 5):
        key, value = _decode_attribute(attr_raw, model_dir, external_files)
        if key:
            attrs[key] = value
    return Node(op_type=op_type, name=name, inputs=inputs, outputs=outputs, attrs=attrs)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def load_model(source) -> Model:
    """Load an ONNX model from a path or raw bytes.

    ``Constant`` nodes are folded into the initializer table at load time so the
    execution engine only ever sees real operators.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        model_dir = "."
    else:
        path = os.fspath(source)
        with open(path, "rb") as handle:
            raw = handle.read()
        model_dir = os.path.dirname(os.path.abspath(path)) or "."

    root = collect(raw)
    ir_version = int(first(root, 1, 0) or 0)
    producer = f"{as_text(first(root, 2))} {as_text(first(root, 3))}".strip()

    opsets: Dict[str, int] = {}
    for raw_opset in all_of(root, 8):
        grouped = collect(raw_opset)
        domain = as_text(first(grouped, 1))
        version = int(first(grouped, 2, 0) or 0)
        opsets[domain] = version

    graph_raw = first(root, 7)
    if graph_raw is None:
        raise OnnxFormatError("file contains no graph")
    graph = collect(graph_raw)
    graph_name = as_text(first(graph, 2))

    external_files: Dict[str, bytes] = {}
    initializers: Dict[str, np.ndarray] = {}
    for init_raw in all_of(graph, 5):
        grouped = collect(init_raw)
        name = as_text(first(grouped, 8))
        initializers[name] = _decode_tensor(init_raw, model_dir, external_files)
        # Older models may leave the initializer out of the inputs list.
        if not name:
            raise OnnxFormatError("initializer without a name")

    nodes: List[Node] = []
    for node_raw in all_of(graph, 1):
        node = _decode_node(node_raw, model_dir, external_files)
        if node.op_type == "Constant":
            # Fold: emit the literal so kernels never see Constant nodes.
            for attr in ("value", "value_float", "value_floats", "value_int",
                         "value_ints", "value_string", "value_strings"):
                if attr in node.attrs:
                    payload = node.attrs[attr]
                    if attr == "value":
                        array = payload
                    elif attr in ("value_float", "value_int", "value_string"):
                        array = np.asarray([payload])
                    else:
                        array = np.asarray(payload)
                    if node.outputs and node.outputs[0]:
                        initializers[node.outputs[0]] = array
                    break
            continue
        nodes.append(node)

    inputs = [_decode_value_info(v) for v in all_of(graph, 11)]
    outputs = [_decode_value_info(v) for v in all_of(graph, 12)]
    value_info = {vi.name: vi for vi in (_decode_value_info(v) for v in all_of(graph, 13))}

    return Model(
        graph_name=graph_name,
        nodes=nodes,
        initializers=initializers,
        inputs=inputs,
        outputs=outputs,
        value_info=value_info,
        opsets=opsets,
        ir_version=ir_version,
        producer=producer,
        model_dir=model_dir,
    )


def describe(model: Model) -> str:
    """Human-readable summary used by the ``inspect`` command."""
    lines = [
        f"graph          : {model.graph_name}",
        f"ir_version     : {model.ir_version}",
        f"producer       : {model.producer or 'unknown'}",
        f"opset          : {model.opsets}",
        f"nodes          : {len(model.nodes)}",
        f"initializers   : {len(model.initializers)}",
    ]
    for value in model.inputs:
        lines.append(f"input          : {value.name} {value.type_name} {value.shape}")
    for value in model.outputs:
        lines.append(f"output         : {value.name} {value.type_name} {value.shape}")
    total = sum(int(v.size) for v in model.initializers.values())
    lines.append(f"weight params  : {total:,}")
    lines.append("operators      :")
    for op, count in sorted(model.op_histogram().items(), key=lambda kv: -kv[1]):
        lines.append(f"    {op:<22} x{count}")
    return "\n".join(lines)
