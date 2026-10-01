# Changelog

## 1.0.0 — 2026-10-01

First release.

### Added
- `yolo-pose devices`: NPU/GPU diagnosis (drivers, device nodes, ONNX Runtime
  execution providers, blockers, and the commands to fix each one).
- Pluggable inference providers through `--provider`
  (`auto`/`npu`/`rocm`/`openvino`/`gpu`/`cpu`/`numpy`); an explicit request fails
  loudly rather than silently degrading to the CPU.
- `yolo-pose camera --pose` / `--no-pose`: pick the pose pass, the face pass, or
  both, instead of always running both.
- Preview window sizing: `--scale` (auto by default, 2x for a 640x480 sensor),
  `--max-window`, plus `--big`/`--huge` shortcuts on the launcher. The window is
  freely resizable and the video follows the drag; the enlargement happens inside
  `XPutImage` (1.4 ms/frame) rather than in NumPy (11.3 ms/frame).
- Dependency-free ONNX parser (protobuf wire format) and NumPy execution engine.
- 60+ operator kernels including Conv/ConvTranspose, MaxPool/AveragePool, Resize,
  GridSample, AffineGrid, BatchNormalization, LRN, and the full elementwise set.
- SiLU fusion (`Sigmoid`+`Mul` → `Silu`) that removes 63 redundant sigmoids from
  the YOLOv8 backbone.
- YOLOv8/YOLO11-pose detection: letterboxing, DFL decoding, NMS, COCO-17 keypoints.
- Anatomical joint-angle analysis (knees, elbows, hips, shoulders, ankles, neck,
  torso) with confidence gating.
- Face pipeline: SCRFD detection with 5 landmarks, Umeyama alignment to the
  ArcFace template, MobileFaceNet embeddings, cosine-similarity gallery matching.
- Pure-NumPy image I/O (PPM/PGM reading, PNG/PNM writing), JPEG decoding, V4L2
  camera capture over mmap, and an X11 preview window.
- CLI: `image`, `camera`, `joints`, `face`, `demo`, `models`, `inspect`, `selftest`.
- Local browser demo (`yolo-pose demo`) with drag-and-drop upload, live pose and
  face rendering, and a preloaded sample gallery.
- Debian packaging for `arch=all` plus native `amd64`/`arm64` builds.
- 99 unit tests (import integrity, ioctl ABI, pixel formats, accelerator
  detection) and a layer-by-layer comparison tool for ONNX Runtime parity.

### Performance
- ONNX Runtime thread count is now tuned by default (capped at 8): 53 ms/frame
  down to 26 ms on the 20-thread reference machine.
- Measured end-to-end on 640x480 camera frames: pose 27.0 ms, face 8.1 ms,
  both 35.1 ms (28.5 FPS). The pure-NumPy engine is 22x slower, so ONNX Runtime
  is used automatically whenever it is importable.
- Concurrent pose+face execution is opt-in (`--parallel`): with ONNX Runtime
  already saturating the CPU it measured slightly *slower* (42 vs 40 ms).
- Note: the AMD XDNA NPU on the reference machine cannot be used yet — the
  published `onnxruntime-vitisai` wheel is aarch64-only, and the x86-64 provider
  ships only inside AMD's full Ryzen AI software package.

### Verified
- YOLOv8n-pose: 282 tensors compared against ONNX Runtime, max abs error 1.6e-3.
- SCRFD-500M: max abs error 4.5e-6. MobileFaceNet: max abs error 2.8e-6.
