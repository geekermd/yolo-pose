# yolo-pose

**YOLO-based human joint (pose) recognition with a dependency-free ONNX runtime.**

`yolo-pose` parses ONNX graphs and executes them with NumPy alone, so the exact same
code runs on **x86-64, aarch64, riscv64** — any platform NumPy supports — with no
compiled wheels, no CUDA, and no `onnxruntime` requirement. When `onnxruntime` *is*
installed it is used automatically for extra speed.

It ships two pipelines:

| Pipeline | Models | Output |
|---|---|---|
| **Body pose** | YOLOv8/YOLO11-pose | 17 COCO keypoints, skeleton, 12 joint angles |
| **Face** | SCRFD + MobileFaceNet | face boxes, 5 landmarks, 512-d embeddings, identity matching |

Everything is pure Python + NumPy, including image I/O, JPEG decoding, V4L2 camera
capture and the X11 preview window.

---

## Highlights

- **No native dependencies.** The ONNX graph parser and every operator kernel
  (Conv, DFL head, GridSample, BatchNormalization, …) are implemented in NumPy.
- **Numerically verified.** All 282 intermediate tensors of YOLOv8n-pose were compared
  against ONNX Runtime; the maximum absolute deviation is **1.6e-3** with zero
  divergent layers. SCRFD and MobileFaceNet match to **4.5e-6** and **2.8e-6**.
- **Runs on ARM.** `arch=all` Debian package plus a build script for native
  `arm64`/`amd64` packages.
- **Two interfaces.** A CLI for everyday use and a clean Python API.
- **Live camera preview** through libX11 (via `ctypes`) with a pure-Python X11
  protocol fallback and a headless image writer.

## Hardware acceleration

`yolo-pose` inspects what the machine actually offers instead of assuming:

```bash
yolo-pose devices          # driver, device nodes, ONNX Runtime EPs, blockers
yolo-pose camera --provider npu     # request a specific execution provider
```

`--provider` accepts `auto` (default), `npu`/`vitis`/`xdna`, `rocm`/`migraphx`,
`openvino`, `gpu`, `cuda`, `cpu`, or `numpy`. An explicitly requested provider
fails loudly instead of silently falling back; `auto` prefers an accelerator and
always keeps the CPU as a fallback.

### Measured on the reference machine (AMD Ryzen AI 9 H 365, 20 threads)

| Path | Pose (640x480) | Face | Both |
|---|---|---|---|
| ONNX Runtime, tuned threads | **27.0 ms** (37 FPS) | **8.1 ms** (124 FPS) | **35.1 ms** (28.5 FPS) |
| Pure NumPy engine | 597.6 ms (1.7 FPS) | — | — |

Two changes produced most of the speed-up:

* **Thread count.** ONNX Runtime's default used only a few of the 20 threads:
  53 ms/frame versus 26 ms with 4 and 24 ms with 8. The default is now capped at
  8 (the measured knee).
* **ONNX Runtime instead of NumPy** when available — a 22× difference here.

Frame preprocessing is *not* a bottleneck for camera-sized input: letterbox plus
normalisation costs 0.54 ms for a 640x480 frame (14 ms for a 1280x720 photo,
which is where the resize actually happens).

### About the AMD XDNA NPU

The reference machine has the hardware and it is ready up to the driver:

* `amdxdna` kernel module loaded, firmware present (`/lib/firmware/amdnpu/1502_00`)
* `/dev/accel/accel0` exists (needs the `render` group: `sudo usermod -aG render $USER`)

What is **not** available is a usable execution provider. `pypi.amd.com` does
publish `onnxruntime-vitisai`, but the only wheel installable on this platform is
built for **aarch64** (`ld-linux-aarch64.so.1`), so it cannot load on x86-64
despite its `py3-none-any` tag. The AMD NPU path needs the full AMD Ryzen AI
software package (developer download, several GB) that ships a matching
`VitisAIExecutionProvider` for x86-64.

`yolo-pose devices` reports this precisely and prints the exact commands to
remedy each blocker. Once such an EP is installed, no code change is needed:
`--provider auto` picks it up.

## Verified performance

YOLOv8n-pose, 640×640, single CPU core class machine (x86-64):

| Backend | Latency | Throughput |
|---|---|---|
| Pure NumPy (SiLU-fused, 210 nodes) | **0.137 s/frame** | 7.3 FPS |
| ONNX Runtime (optional) | **0.033 s/frame** | 30 FPS |

---

## Install

### Debian / Ubuntu package (recommended)

```bash
sudo apt install ./yolo-pose_1.0.0_all.deb
yolo-pose models pull --face        # optional: download the face models
```

The package depends only on `python3-numpy`; `python3-pil` (recommended) and
`python3-onnxruntime` (optional, for speed) are picked up automatically when present.

### From source

```bash
pip install .            # add [fast] for onnxruntime, [pil] for Pillow
yolo-pose models pull    # downloads models/yolov8n-pose.onnx (~13 MB)
```

### Build a package yourself

```bash
./debian/build-deb.sh                  # arch-independent (all)
./debian/build-deb.sh --arch amd64     # native amd64 package
./debian/build-deb.sh --arch arm64     # native aarch64 package (needs qemu/dpkg)
```

---

## Usage

### Body pose

```bash
# annotate one image
yolo-pose image photo.jpg -o annotated.png

# print every keypoint and joint angle as JSON
yolo-pose joints photo.jpg --json -

# live camera preview (press q or Esc to quit)
yolo-pose camera --device 0
yolo-pose camera --device 0 --faces --gallery people.npz     # pose + face in one pass
yolo-pose camera --device 0 --faces --conf 1.01              # face only
yolo-pose camera --device 0 --no-window --save-dir frames/   # headless
yolo-pose camera --list                                      # probe devices
yolo-pose devices                                            # NPU/GPU diagnosis
yolo-pose camera --pose                                      # pose only
yolo-pose camera --no-pose --faces                           # face only
yolo-pose camera --faces --provider auto                     # try an accelerator

# verify the installation
yolo-pose selftest
```

### Face recognition

```bash
# 1. download the face models (~16 MB)
yolo-pose models pull --face

# 2. enrol people from labelled photos into a gallery
yolo-pose face alice_1.jpg --enroll alice --gallery people.npz --save-gallery people.npz
yolo-pose face bob_1.jpg   --enroll bob   --gallery people.npz --save-gallery people.npz

# 3. recognise
yolo-pose face group.jpg --gallery people.npz -o annotated.png
yolo-pose face group.jpg --gallery people.npz --json - --threshold 0.35
```

### Preview window size and frame rate

The recognition window opens **large and locked**: 2x by default, so a 640x480
sensor fills a 1280x960 window. It stays that size unless you ask otherwise.
Frames are still captured and inferred at the sensor's native resolution — only
the display is enlarged, so a big window costs no accuracy.

```bash
yolo-pose camera --scale 0        # auto (default): fits the screen, 2x for 640x480
yolo-pose camera --scale 3        # 3x -> 1920x1440
yolo-pose camera --scale 1.5      # any factor
yolo-pose camera --resizable      # allow dragging the window edges
yolo-pose camera --max-window 2200
```

The enlargement happens in NumPy with nearest-neighbour replication: **1.9 ms**
for 2x (3x costs 4.3 ms). It is deliberately *not* left to ``XPutImage``, which
Xwayland treats as a clipped blit rather than a stretch — that mismatch showed up
as a small picture sitting in the corner of a large window. A regression test
asserts that the enlarged frame's corners land in the window's corners.

#### Making it faster

The display and the model are **decoupled by default**: inference runs on a
background thread against a latest-only frame slot, so the preview always draws
the newest frame at the camera's own rate while the overlay updates as results
arrive. Use `--sync` to make the preview wait for each inference instead.

Measured end-to-end on 640x480 -> a 1280x960 window, AMD Ryzen AI 9 H 365:

| Mode | Display frame rate |
|---|---|
| `--sync` (preview waits for the model) | 21.1 FPS |
| **async (default)** | **28.2 FPS** — the camera itself tops out at 29.5 FPS |

Inference costs ~34 ms per frame, so anything that waits on it cannot exceed
~29 FPS; running it asynchronously removes that coupling entirely.

What was measured and rejected:

| Idea | Result |
|---|---|
| More ONNX Runtime threads | Worse: 8 threads 30.5 ms, 20 threads 59.6 ms |
| CPU pinning to the 5.09 GHz cores | No gain (33.1 ms vs 30.5 ms) |
| Many single-thread sessions in parallel | Only 30-35 FPS total throughput |
| INT8 dynamic quantisation | 3x *slower* (110 ms vs 34 ms) |
| FP16 weights | 9% faster (30.5 ms vs 33.6 ms) |
| JPEG capture (`MJPG`) | Only helps above 640x480; YUYV is fine at 640x480 |

The hard limit is computation: YOLOv8n-pose needs ~30 ms per 640x640 frame on
this CPU. Going meaningfully beyond ~30 FPS needs a smaller input resolution
(re-export), which is a model-side change rather than a runtime one.

* **Thread count matters more than anything else.** ONNX Runtime's default used
  only a few of 20 threads (53 ms/frame for pose alone). A sweep showed 8 threads
  optimal (30.5 ms), with 20 threads *worse* (59.6 ms). The default is now capped
  at 8.
* **Smallest model wins.** Quantised variants gave little: fp16 26.0 ms vs fp32
  28.1 ms, q4 29.2 ms, and int8 does not load (needs quantised operators the NumPy
  engine does not implement).
* `--detect-every N` runs inference every Nth frame and redraws the previous
  detections on the frames in between, so the display stays smooth.
* Pose alone reaches the camera's 30 FPS ceiling; adding the face pipeline costs
  about 20 ms, so use `--pose` when only joints are needed.

### One-click launchers

```bash
./run-camera.sh              # live window: pose + face recognition
./run-camera.sh --pose-only  # pose only
./run-camera.sh --face-only  # face only
./run-camera.sh --save       # also record to frames/session.mp4
./run-camera.sh --big        # start with a 2x window (1280x960)
./run-camera.sh --huge       # start with a 3x window (1920x1440)
./run-camera.sh --scale 1.5  # any factor; still resizable by dragging
./run-demo.sh --open         # browser demo, opens automatically
./camera-menu.sh             # ask which mode to start (zenity dialog)
```

The launchers are also installed as desktop/menu entries
(`launchers/*.desktop` → `~/.local/share/applications/`), so you can pick the
mode from the application menu or the taskbar instead of the shell.

### Camera backends

Capture uses the native V4L2 ioctl path (mmap streaming, no external process) and
falls back to driving `ffmpeg` when the kernel ioctls needed to negotiate a
format are unavailable — for example inside a sandbox that filters
`VIDIOC_G_FMT`/`VIDIOC_S_FMT`. Pick one explicitly with
`--capture-backend v4l2|ffmpeg|auto`; `auto` prefers the native path, which is
roughly an order of magnitude faster to start.

If frames come out black, the lens is covered or the camera has no light — the
pipeline itself is running (the HUD shows the FPS and detection counts).

### Local browser demo

```bash
yolo-pose demo                   # then open http://127.0.0.1:8765
yolo-pose demo --open            # start and open the browser automatically
yolo-pose demo --port 9000       # different port
```

The demo binds to `127.0.0.1` only. Drop in any photo (or click "载入样例图" for the
bundled sample), pick pose / face / both, and see annotated results plus a JSON
breakdown. It ships a small gallery (`examples/demo_gallery.npz`) so face
recognition shows real names out of the box.

### Models

```bash
yolo-pose models list            # everything, with install status
yolo-pose models pull            # default pose model
yolo-pose models pull --face     # SCRFD detector + MobileFaceNet embedder
yolo-pose models pull yolov8s-pose
yolo-pose inspect                # graph, opset and operator coverage
```

### Python API

```python
from yolopose import Detector
from yolopose.vision.imaging import load_image
from yolopose.render.skeleton import draw_pose

detector = Detector("models/yolov8n-pose.onnx", backend="auto")
image = load_image("photo.jpg")
result = detector.detect(image)

for pose, joints in zip(result.poses, result.joints):
    print(pose.score, joints.angles)

canvas = image.copy()
for pose in result.poses:
    draw_pose(canvas, pose)
```

```python
from yolopose.face import FaceGallery, FaceRecognizer
from yolopose.vision.imaging import load_image

recognizer = FaceRecognizer(gallery=FaceGallery.load("people.npz"))
result = recognizer.recognize(load_image("group.jpg"))
for face in result.faces:
    print(face.name, face.similarity)
```

---

## Architecture

```
src/yolopose/
├── core/
│   ├── onnx_graph.py    # protobuf wire-format ONNX parser (no protobuf needed)
│   └── runtime.py       # NumPy operator kernels + graph executor
├── vision/
│   ├── detector.py      # backend-agnostic pose detector
│   ├── pose.py          # DFL decoding, NMS, COCO-17 keypoints
│   ├── joints.py        # joint-angle computation
│   ├── imaging.py       # letterbox, resize, PNM/PNG read+write
│   ├── jpeg.py          # JPEG decoding (Pillow preferred, builtin fallback)
│   └── camera.py        # V4L2 capture over mmap
├── face/
│   ├── scrfd.py         # SCRFD anchor decoding
│   ├── align.py         # Umeyama alignment, embeddings
│   └── recognizer.py    # gallery, matching, rendering
├── render/
│   ├── skeleton.py      # anti-aliased drawing + built-in 5x7 font
│   ├── viewer.py        # Xlib (ctypes) viewer, headless fallback
│   └── x11.py           # pure-Python X11 protocol viewer
└── cli.py               # command line interface
```

The runtime fuses `Sigmoid` + `Mul` into a single `Silu` kernel, which removes 63
redundant sigmoid evaluations from the YOLOv8 backbone.

## Joint angles

Angles are reported in degrees, measured with the joint itself as the vertex:

| Key | Meaning | Straight limb |
|---|---|---|
| `left/right_knee` | thigh vs shin | 180° |
| `left/right_elbow` | upper arm vs forearm | 180° |
| `left/right_hip` | torso vs thigh | 180° |
| `left/right_shoulder` | torso vs upper arm | 90° (arms down) |
| `left/right_ankle` | shin vs virtual toe | 180° |
| `neck` | head axis vs body axis | — |
| `torso` | shoulder line vs body line | — |

An angle is only reported when every contributing keypoint clears
`--kpt-conf` (default 0.5); otherwise it appears under `skipped` with a reason.

## Models and licensing

**The source code of this project is MIT licensed. The pretrained model weights are
not** — they come from Ultralytics and InsightFace and carry their own licences:

| Model | Source | Licence |
|---|---|---|
| `yolov8n-pose.onnx` | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| `scrfd_500m.onnx` | [InsightFace](https://github.com/deepinsight/insightface) | MIT (code) / non-commercial research |
| `w600k_mbf.onnx` | [InsightFace](https://github.com/deepinsight/insightface) | MIT (code) / non-commercial research |

Weights are **downloaded at install time**, never committed to this repository, so
the MIT licence of the code is unaffected. See [NOTICE](NOTICE) for details and
check the upstream licences before any commercial use.

Model downloads default to the `hf-mirror.com` mirror and fall back to
`huggingface.co`, so they work on networks where Hugging Face is unreachable.

## Building and testing

```bash
python -m pytest tests/ -q          # 98 tests, <1 s
yolo-pose selftest                  # end-to-end check on your machine
python tools/compare_intermediates.py   # layer-by-layer check against onnxruntime
```

## Requirements

- Python ≥ 3.8, NumPy ≥ 1.20
- Optional: Pillow (better JPEG/PNG), ONNX Runtime (≈4× faster), ffmpeg (`--record`)
- Linux for camera capture (V4L2) and the X11 viewer; detection itself is portable

## Known limitations

- The preview window needs an X server. On Wayland, XWayland works; otherwise use
  `--no-window --save-dir`.
- The built-in JPEG decoder (`yolopose.vision.jpeg.decode_jpeg_builtin`) is
  experimental for colour streams; install `python3-pil` for production MJPEG use.
- CPU only. `--gpu` is accepted but ignored.
- Video input is handled through `--record` (ffmpeg); direct video-file decoding is
  not implemented.

## License

MIT — see [LICENSE](LICENSE). Third-party model licences are described in [NOTICE](NOTICE).
