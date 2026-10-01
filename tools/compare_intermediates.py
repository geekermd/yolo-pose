"""Layer-by-layer numerical comparison: NumPy engine vs ONNX Runtime.

Runs both engines on the same graph, exposing every intermediate tensor as a
session output, and reports the first node whose output diverges.
"""
import sys
sys.path.insert(0, "/tmp/yolo-pose/src")

import numpy as np
import onnx
import onnxruntime as ort

from yolopose.core.runtime import NumpyEngine
from yolopose.core import runtime as R

MODEL = "/tmp/yp-build/yolov8n-pose-fp32.onnx"


def make_probe_model(path, keep_ops):
    """Return a model whose outputs include the outputs of `keep_ops`."""
    m = onnx.load(path)
    infer = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    types = {}
    for vi in list(infer.graph.value_info) + list(infer.graph.output):
        types[vi.name] = vi.type.tensor_type.elem_type
    existing = {o.name for o in m.graph.output}
    for node in m.graph.node:
        # skip shape-manipulation outputs whose dtype is int64 (Shape/Gather/...)
        for out in node.output:
            if not out or out in existing:
                continue
            elem = types.get(out)
            if elem is None:
                continue
            m.graph.output.extend([
                onnx.helper.make_tensor_value_info(out, elem, None)
            ])
            existing.add(out)
    return m


def main():
    x = np.random.RandomState(0).rand(1, 3, 640, 640).astype(np.float32)

    # 1) run the numpy engine while recording every tensor it produces
    eng = NumpyEngine(MODEL, fuse=False)
    env = dict(eng.weights)
    env[eng.input_name] = x
    produced = {}
    for node in eng.nodes:
        result = R.OP_TABLE[node.op_type](node, env)
        outs = result if isinstance(result, (list, tuple)) else [result]
        for name, value in zip(node.outputs, outs):
            env[name] = value
            produced[name] = np.asarray(value)
    print(f"numpy engine produced {len(produced)} intermediate tensors")

    # 2) ask onnxruntime for the same tensors
    probe = make_probe_model(MODEL, {n.op_type for n in eng.nodes})
    probe_path = "/tmp/yp-build/_probe.onnx"
    onnx.save(probe, probe_path)
    sess = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
    wanted = [o.name for o in sess.get_outputs() if o.name in produced]
    print(f"onnxruntime exposing {len(wanted)} comparable tensors")
    refs = dict(zip(wanted, sess.run(wanted, {"images": x})))

    # 3) compare and locate divergence
    rows = []
    for name in wanted:
        mine = produced[name]
        ref = refs[name]
        if mine.shape != ref.shape:
            rows.append((name, f"SHAPE {mine.shape} vs {ref.shape}", np.inf, np.inf))
            continue
        diff = np.abs(mine.astype(np.float64) - ref.astype(np.float64))
        scale = np.maximum(np.abs(ref.astype(np.float64)), 1e-3)
        rows.append((name, "", diff.max(), (diff / scale).max()))

    order = {n: i for i, n in enumerate(produced)}
    rows.sort(key=lambda r: order.get(r[0], 0))
    bad = 0
    for name, note, max_abs, max_rel in rows:
        flag = ""
        if note or max_abs > 1e-2:
            flag = "  <-- DIVERGES"
            bad += 1
        if flag or max_abs > 1e-4:
            print(f"  {name:52} max_abs={max_abs:10.3e} max_rel={max_rel:10.3e}{note}{flag}")
    print(f"\ntensors compared={len(rows)}  diverging(>1e-2)={bad}")

    # first diverging node in execution order
    for name, note, max_abs, _ in rows:
        if note or max_abs > 1e-2:
            for node in eng.nodes:
                if name in node.output:
                    print(f"\nFIRST DIVERGENCE at op={node.op_type} node={node.name}")
                    print(f"  inputs={node.inputs}")
                    print(f"  attrs={node.attrs}")
                    for inp in node.inputs:
                        if inp in produced:
                            print(f"    input {inp!r} shape={produced[inp].shape}")
                    break
            break


if __name__ == "__main__":
    main()
