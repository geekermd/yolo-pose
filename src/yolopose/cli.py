"""Command line interface for yolo-pose."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

from . import __version__
from .models import (DEFAULT_MODEL, FACE_MODELS, MODELS, find_model,
                     list_models, pull_model)
from .render.skeleton import (draw_angle_labels, draw_hud, draw_pose)
from .render.viewer import ViewerError, open_viewer
from .vision.detector import Detector, available_backends
from .vision.imaging import load_image, save_image
from .vision.pose import KEYPOINT_NAMES

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_MODEL = 2


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo-pose",
        description="Human joint (pose) recognition with YOLO, running on a "
                    "pure-NumPy ONNX runtime for x86-64 and ARM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  yolo-pose models pull                      download the default model
  yolo-pose image photo.jpg -o out.png       detect and annotate one image
  yolo-pose camera --device 0                live preview from a webcam
  yolo-pose joints photo.jpg --json          print measured joint angles
  yolo-pose inspect model.onnx               show graph and operator coverage
""")
    parser.add_argument("--version", action="version",
                        version=f"yolo-pose {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, default_model: bool = True):
        sub.add_argument("-m", "--model", default=None,
                         help="path to a pose .onnx model, or a model name "
                              f"({', '.join(MODELS)})")
        sub.add_argument("--backend", default="auto",
                         choices=["auto", "numpy", "onnxruntime"],
                         help="inference backend (default: auto)")
        sub.add_argument("--conf", type=float, default=0.25,
                         help="person confidence threshold (default: 0.25)")
        sub.add_argument("--iou", type=float, default=0.45,
                         help="NMS IoU threshold (default: 0.45)")
        sub.add_argument("--kpt-conf", type=float, default=0.5,
                         help="keypoint visibility threshold (default: 0.5)")
        sub.add_argument("--max-det", type=int, default=100,
                         help="maximum persons per frame (default: 100)")
        sub.add_argument("--threads", type=int, default=0,
                         help="ONNX Runtime intra-op threads (0 = library default)")
        sub.add_argument("-v", "--verbose", action="store_true")

    # -- image -------------------------------------------------------------- #
    p_image = subparsers.add_parser("image", help="annotate a single image")
    p_image.add_argument("input", nargs="+", help="input image file(s)")
    p_image.add_argument("-o", "--output", default=None,
                         help="output path (single input) or directory")
    p_image.add_argument("--json", dest="json_path", default=None,
                         help="also write detections as JSON to this path")
    p_image.add_argument("--no-box", action="store_true", help="hide person boxes")
    p_image.add_argument("--no-angles", action="store_true",
                         help="hide joint angle labels")
    p_image.add_argument("--line-width", type=float, default=2.0)
    add_common(p_image)

    # -- camera ------------------------------------------------------------- #
    p_cam = subparsers.add_parser("camera", help="live preview from a camera")
    p_cam.add_argument("-d", "--device", default="0",
                       help="camera device, e.g. 0 or /dev/video0 (default: 0)")
    p_cam.add_argument("--format", default="auto",
                       help="pixel format: auto, YUYV, MJPG, RGB3 (default: auto)")
    p_cam.add_argument("--width", type=int, default=640)
    p_cam.add_argument("--height", type=int, default=480)
    p_cam.add_argument("--fps", type=int, default=30)
    p_cam.add_argument("--profile", default="accurate",
                       choices=["accurate", "balanced", "fast", "turbo"],
                       help="pose model resolution preset: accurate=640, "
                            "balanced=416, fast=320 (2.3x faster), turbo=256")
    p_cam.add_argument("--scale", type=float, default=0.0,
                       help="enlarge the preview window by this factor "
                            "(default: auto — fits your screen)")
    p_cam.add_argument("--max-window", type=int, default=1600,
                       help="upper bound for the preview window width in pixels "
                            "(default: 1600)")
    p_cam.add_argument("--resizable", action="store_true",
                       help="allow dragging the window edges (default: locked)")
    p_cam.add_argument("--async", dest="async_infer", action="store_true",
                       default=None,
                       help="run inference on a background thread so the preview "
                            "always shows the newest frame (default: on)")
    p_cam.add_argument("--sync", dest="async_infer", action="store_false",
                       help="block the preview on every inference")
    p_cam.add_argument("--detect-every", type=int, default=1, metavar="N",
                       help="run inference every Nth frame and reuse the last "
                            "annotations in between (1 = every frame). Raises the "
                            "display frame rate when inference is the bottleneck.")
    p_cam.add_argument("--capture-format", default=None,
                       choices=["YUYV", "MJPG", "RGB3", "BGR3"],
                       help="force a pixel format. MJPG sustains 30 FPS at "
                            "1280x720 where YUYV drops to ~9 FPS (USB bandwidth).")
    p_cam.add_argument("--frames", type=int, default=0,
                       help="stop after N frames (0 = until closed)")
    p_cam.add_argument("--no-window", action="store_true",
                       help="headless: write annotated frames to --save-dir")
    p_cam.add_argument("--save-dir", default=None,
                       help="directory for headless frame output")
    p_cam.add_argument("--save-every", type=int, default=10,
                       help="save every Nth frame in headless mode")
    p_cam.add_argument("--record", default=None,
                       help="write the annotated preview to this video file "
                            "(requires ffmpeg)")
    p_cam.add_argument("--list", action="store_true",
                       help="list capture devices and exit")
    p_cam.add_argument("--capture-backend", default="auto",
                       choices=["auto", "v4l2", "ffmpeg"],
                       help="capture backend: v4l2 (native mmap) or ffmpeg "
                            "(used automatically when the native ioctls are "
                            "blocked)")
    p_cam.add_argument("--faces", action="store_true",
                       help="also run face detection/recognition each frame")
    p_cam.add_argument("--pose", dest="pose", action="store_true", default=None,
                       help="force the pose pass on")
    p_cam.add_argument("--no-pose", dest="pose", action="store_false",
                       help="skip the pose pass (implies --faces)")
    p_cam.add_argument("--parallel", dest="parallel", action="store_true",
                       default=None,
                       help="run pose and face concurrently (usually slower: the "
                            "ONNX Runtime sessions already saturate the CPU)")
    p_cam.add_argument("--no-parallel", dest="parallel", action="store_false",
                       help="run pose then face sequentially")
    p_cam.add_argument("--gallery", default=None,
                       help="face gallery (.npz/.json) for identity labels")
    p_cam.add_argument("--face-score", type=float, default=0.5,
                       help="face detection threshold (default: 0.5)")
    p_cam.add_argument("--threshold", type=float, default=0.35,
                       help="face match threshold (default: 0.35)")
    add_common(p_cam)

    # -- joints ------------------------------------------------------------- #
    p_joints = subparsers.add_parser(
        "joints", help="print measured joint angles (no image output)")
    p_joints.add_argument("input", nargs="+", help="input image file(s)")
    p_joints.add_argument("--json", dest="json_path", default=None,
                          help="write results as JSON (use '-' for stdout)")
    add_common(p_joints)

    # -- face --------------------------------------------------------------- #
    p_face = subparsers.add_parser(
        "face", help="face detection and recognition (SCRFD + embeddings)")
    p_face.add_argument("input", nargs="*", help="input image file(s)")
    p_face.add_argument("-o", "--output", default=None,
                        help="annotated output path (single input) or directory")
    p_face.add_argument("--json", dest="json_path", default=None,
                        help="write results as JSON ('-' for stdout)")
    p_face.add_argument("--gallery", default=None,
                        help="gallery file (.npz or .json) with known faces")
    p_face.add_argument("--enroll", metavar="NAME", default=None,
                        help="add detected faces from the input to the gallery "
                             "under NAME instead of recognising")
    p_face.add_argument("--save-gallery", default=None,
                        help="write the updated gallery to this path")
    p_face.add_argument("--threshold", type=float, default=0.35,
                        help="cosine similarity threshold for a match "
                             "(default: 0.35)")
    p_face.add_argument("--face-score", type=float, default=0.5,
                        help="face detection confidence threshold (default: 0.5)")
    p_face.add_argument("--face-nms", type=float, default=0.4,
                        help="face detection NMS IoU (default: 0.4)")
    p_face.add_argument("--det-model", default=None,
                        help="SCRFD detector .onnx (or a registry name)")
    p_face.add_argument("--emb-model", default=None,
                        help="face embedding .onnx (or a registry name)")
    p_face.add_argument("--no-align", action="store_true",
                        help="use the raw face box instead of the aligned crop")
    p_face.add_argument("--save-crops", default=None,
                        help="directory to write aligned 112x112 face crops")
    p_face.add_argument("--no-landmarks", action="store_true")
    p_face.add_argument("--gpu", action="store_true",
                        help="accepted for compatibility; CPU only")
    p_face.add_argument("--backend", default="auto",
                        choices=["auto", "numpy", "onnxruntime"])
    p_face.add_argument("--threads", type=int, default=0)
    p_face.add_argument("-v", "--verbose", action="store_true")

    # -- demo --------------------------------------------------------------- #
    p_demo = subparsers.add_parser(
        "demo", help="start a local browser demo (upload an image, see results)")
    p_demo.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: 127.0.0.1, local only)")
    p_demo.add_argument("--port", type=int, default=8765,
                        help="port (default: 8765)")
    p_demo.add_argument("-m", "--model", default=None,
                        help="pose model path or registry name")
    p_demo.add_argument("--det-model", default=None, help="SCRFD detector model")
    p_demo.add_argument("--emb-model", default=None, help="face embedder model")
    p_demo.add_argument("--backend", default="auto",
                        choices=["auto", "numpy", "onnxruntime"])
    p_demo.add_argument("--open", action="store_true",
                        help="open the page in the default browser")
    p_demo.add_argument("--strict-port", action="store_true",
                        help="fail instead of reusing/advancing the port")

    # -- devices ------------------------------------------------------------ #
    p_dev = subparsers.add_parser(
        "devices", help="diagnose NPU/GPU availability and execution providers")
    p_dev.add_argument("--json", action="store_true", help="emit raw JSON")

    # -- models ------------------------------------------------------------- #
    p_models = subparsers.add_parser("models", help="list or download models")
    model_sub = p_models.add_subparsers(dest="models_command", required=True)
    p_list = model_sub.add_parser("list", help="list known models")
    p_list.add_argument("--json", action="store_true")
    p_pull = model_sub.add_parser("pull", help="download a model")
    p_pull.add_argument("name", nargs="?", default=DEFAULT_MODEL)
    p_pull.add_argument("-o", "--output", default=None)
    p_pull.add_argument("-f", "--force", action="store_true")
    p_pull.add_argument("--all", action="store_true", help="download every model")
    p_pull.add_argument("--face", action="store_true",
                        help="download the face pipeline models instead")

    # -- inspect ------------------------------------------------------------ #
    p_inspect = subparsers.add_parser(
        "inspect", help="show an ONNX graph, opset and operator coverage")
    p_inspect.add_argument("model", nargs="?", default=None,
                           help="model path (defaults to the installed model)")
    p_inspect.add_argument("--ops", action="store_true",
                           help="list every node in the graph")

    # -- selftest ----------------------------------------------------------- #
    p_test = subparsers.add_parser(
        "selftest", help="verify the installation end to end")
    p_test.add_argument("--model", default=None)

    return parser


# Pose input-resolution presets.  Speeds measured on the reference machine for
# the full detect() call; accuracy loss is relative to the 640 model.
POSE_PROFILES = {
    "accurate": (640, "baseline accuracy"),
    "balanced": (416, "~1.4x faster, accurate"),
    "fast": (320, "~2.3x faster, main people still detected"),
    "turbo": (256, "~3.4x faster, noticeably less precise"),
}


def _profile_model(requested: Optional[str], profile: str) -> Optional[str]:
    """Map a profile name onto a converted model, when one is installed.

    Falls back to the caller's model (or the default) when the sized variant is
    not present, so the presets never break a working setup.
    """
    if requested:
        return requested
    if not profile or profile == "accurate":
        return None
    size = POSE_PROFILES.get(profile, (640, ""))[0]
    candidate = find_model(f"yolov8n-pose-{size}")
    return candidate or None


def _resolve_model(requested: Optional[str]) -> str:
    path = find_model(requested)
    if path is None:
        raise SystemExit(
            f"no pose model found (looked for {requested or DEFAULT_MODEL}).\n"
            "Run `yolo-pose models pull` to download it."
        )
    return path


def _make_detector(args) -> Detector:
    requested = getattr(args, "model", None)
    profile = getattr(args, "profile", "accurate")
    model_path = _resolve_model(_profile_model(requested, profile) or requested)
    return Detector(
        model_path,
        backend=getattr(args, "backend", "auto"),
        conf_threshold=getattr(args, "conf", 0.25),
        iou_threshold=getattr(args, "iou", 0.45),
        max_detections=getattr(args, "max_det", 100),
        threads=getattr(args, "threads", 0),
        verbose=getattr(args, "verbose", False),
    )


def _annotate(image: np.ndarray, result, show_box: bool = True,
              show_angles: bool = True, line_width: float = 2.0) -> np.ndarray:
    canvas = np.ascontiguousarray(image.copy())
    for pose, joints in zip(result.poses, result.joints):
        draw_pose(canvas, pose, show_box=show_box, show_score=True)
        if show_angles:
            draw_angle_labels(canvas, pose, joints, scale=1)
    return canvas


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def command_image(args) -> int:
    detector = _make_detector(args)
    single = len(args.input) == 1
    output = args.output
    if not single and output and not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)

    all_results = {}
    for path in args.input:
        image, result = detector.detect_file(path, keypoint_threshold=args.kpt_conf)
        canvas = _annotate(image, result, show_box=not args.no_box,
                           show_angles=not args.no_angles,
                           line_width=args.line_width)

        if output is None:
            target = f"{os.path.splitext(path)[0]}_pose.png"
        elif single:
            target = output
        else:
            target = os.path.join(output, os.path.basename(path))
            target = os.path.splitext(target)[0] + "_pose.png"
        save_image(target, canvas)

        print(f"{path}: {len(result.poses)} person(s) in "
              f"{result.inference_seconds * 1000:.0f} ms "
              f"({result.backend}) -> {target}")
        for index, (pose, joints) in enumerate(zip(result.poses, result.joints), 1):
            visible = int(np.count_nonzero(pose.keypoint_scores >= args.kpt_conf))
            print(f"  person {index}: score={pose.score:.3f} "
                  f"visible_keypoints={visible}/17")
            if joints.angles:
                summary = ", ".join(f"{k}={v:.1f}" for k, v in
                                    sorted(joints.angles.items()))
                print(f"    joint angles: {summary}")
        all_results[path] = result.as_dict()

    if args.json_path:
        payload = json.dumps(all_results, indent=2, ensure_ascii=False)
        if args.json_path == "-":
            print(payload)
        else:
            with open(args.json_path, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            print(f"wrote {args.json_path}")
    return EXIT_OK


def command_joints(args) -> int:
    detector = _make_detector(args)
    payload = {}
    status = EXIT_OK
    for path in args.input:
        _, result = detector.detect_file(path, keypoint_threshold=args.kpt_conf)
        if not result.poses:
            status = EXIT_ERROR
        print(f"{path}: {len(result.poses)} person(s)")
        for index, (pose, joints) in enumerate(zip(result.poses, result.joints), 1):
            print(f"  person {index}  score={pose.score:.3f}  "
                  f"box=[{pose.box[0]:.0f},{pose.box[1]:.0f},"
                  f"{pose.box[2]:.0f},{pose.box[3]:.0f}]")
            for name in KEYPOINT_NAMES:
                x, y, score = pose.keypoint(name)
                marker = "*" if score >= args.kpt_conf else " "
                print(f"    {marker} {name:<15} x={x:7.1f} y={y:7.1f} conf={score:.3f}")
            if joints.angles:
                print("    angles:")
                for name, value in sorted(joints.angles.items()):
                    print(f"      {name:<16} {value:6.1f}°")
            if joints.skipped:
                print(f"    skipped: {', '.join(sorted(joints.skipped))}")
        payload[path] = result.as_dict()

    if args.json_path:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.json_path == "-":
            print(text)
        else:
            with open(args.json_path, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
            print(f"wrote {args.json_path}")
    return status if status == EXIT_OK or payload else EXIT_ERROR


def _start_recorder(path: str, width: int, height: int, fps: int):
    """Start an ffmpeg subprocess that accepts raw RGB frames on stdin."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("warning: ffmpeg not found; --record is disabled", file=sys.stderr)
        return None
    command = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-pix_fmt", "yuv420p", "-vcodec", "libx264", path]
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE)
    except Exception as exc:
        print(f"warning: could not start ffmpeg ({exc}); --record disabled",
              file=sys.stderr)
        return None


def _resolve_display_scale(requested: float, max_window: int,
                           width: int, height: int) -> float:
    """Pick a preview magnification factor.

    ``requested > 0`` is honoured (clamped to a sane range).  Otherwise the
    largest integer-ish factor that keeps the window within ``max_window`` is
    chosen, so a 640x480 sensor gets a window around 1280x960 by default.
    """
    if width <= 0 or height <= 0:
        return 1.0
    if requested and requested > 0:
        return max(0.25, min(float(requested), 8.0))

    limit = max(320, int(max_window))
    if width >= limit:
        return 1.0
    factor = limit / float(width)
    # Prefer whole-number magnification, which keeps the image crisp.
    whole = int(factor)
    scale = float(whole) if whole >= 2 else factor
    return max(1.0, min(scale, 4.0))


def _viewer_accepts_window_size(viewer) -> bool:
    """True when a viewer's ``show`` takes an explicit window size."""
    try:
        import inspect
        parameters = inspect.signature(viewer.show).parameters
        return "window_size" in parameters
    except (TypeError, ValueError):
        return False


def _scale_for_display(canvas: np.ndarray, scale: float) -> np.ndarray:
    """Enlarge a frame for on-screen display.

    Integer factors use nearest-neighbour replication: it is exact, keeps the
    skeleton pixel-crisp, and costs ~1.9 ms for a 2x enlargement (a bilinear
    resize of the same frame measured 55 ms).  Fractional factors fall back to
    bilinear resampling.
    """
    if scale <= 1.0:
        return canvas
    whole = int(round(scale))
    if abs(scale - whole) < 1e-6 and whole >= 2:
        return np.ascontiguousarray(
            canvas.repeat(whole, axis=0).repeat(whole, axis=1))
    from .vision.imaging import resize_bilinear
    height = max(1, int(round(canvas.shape[0] * scale)))
    width = max(1, int(round(canvas.shape[1] * scale)))
    return resize_bilinear(canvas, (height, width))


def command_camera(args) -> int:
    """Live pose (and optionally face) recognition from a camera."""
    from .vision.camera import CameraError, capture_devices, list_cameras
    from .vision.capture import backend_name, camera_backends, open_any_camera

    if args.list:
        devices = list_cameras()
        if not devices:
            print("no /dev/video* nodes found")
            return EXIT_ERROR
        found = dict(capture_devices())
        for device in devices:
            if device in found:
                info = found[device]
                print(f"{device}: {info['card']}  formats={info['formats']}")
            else:
                print(f"{device}: (not a video capture device)")
        print(f"available backends: {', '.join(camera_backends())}")
        return EXIT_OK

    # Decide which passes run.  Explicit flags win; otherwise a person threshold
    # above 1.0 (or --no-pose) disables the pose pass, and --faces enables faces.
    run_faces = bool(args.faces)
    if args.pose is False:
        run_pose = False
        run_faces = True          # "no pose" only makes sense with faces
    elif args.pose is True:
        run_pose = True
    else:
        run_pose = args.conf <= 1.0
    if not run_pose and not run_faces:
        run_faces = True          # never leave the loop with nothing to do

    detector = _make_detector(args) if run_pose else None

    recognizer = None
    if run_faces:
        try:
            from .face.recognizer import FaceGallery, FaceRecognizer

            gallery = None
            if args.gallery:
                gallery = FaceGallery.load(args.gallery, threshold=args.threshold)
                print(f"gallery: {len(gallery)} entries, {len(gallery.people)} people")
            recognizer = FaceRecognizer(backend=args.backend, gallery=gallery,
                                        threshold=args.threshold,
                                        verbose=args.verbose)
        except FileNotFoundError as exc:
            print(f"warning: face pipeline unavailable ({exc})", file=sys.stderr)
            recognizer = None

    # Pose and face are independent pipelines, so overlap them instead of paying
    # the sum of both latencies.  Each uses its own ONNX Runtime session and the
    # kernels release the GIL, so threads genuinely help.
    # Measured on the reference machine: with ONNX Runtime already using 8
    # intra-op threads, running the two pipelines concurrently was *slower*
    # (42 ms/frame versus 40 ms) because they only contend for the same cores.
    # So parallelism is opt-in and defaults off; it can still help on machines
    # where each session is thread-limited.
    run_parallel = (detector is not None and recognizer is not None
                    and args.parallel is True)
    executor = None
    if run_parallel:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=2,
                                      thread_name_prefix="yolo-pose")

    requested_format = args.capture_format or args.format
    try:
        camera = open_any_camera(args.device, args.width, args.height, args.fps,
                                 requested_format,
                                 backend=args.capture_backend)
    except CameraError as exc:
        print(f"camera error: {exc}", file=sys.stderr)
        if "not installed" in str(exc) or "no camera backend" in str(exc):
            print("hint: `yolo-pose camera --list` shows usable devices; "
                  "install ffmpeg for the fallback backend", file=sys.stderr)
        return EXIT_ERROR

    backend = backend_name(camera)

    # Frames are captured and inferred at their native size; only the *displayed*
    # image is enlarged, so a small sensor still gives a big, readable window
    # without any loss of detection accuracy.
    native_w, native_h = camera.format.width, camera.format.height
    display_scale = _resolve_display_scale(args.scale, args.max_window,
                                           native_w, native_h)
    window_size = (int(native_w * display_scale), int(native_h * display_scale))
    # When the window is larger than the frame, prefer letting the display
    # backend stretch it during the blit: a NumPy resize costs ~2.3 ms per frame
    # for 2x, while XPutImage scaling is free.

    viewer = None
    display = "none"
    recorder = None
    frames_written = 0

    try:
        if not args.no_window:
            try:
                viewer, display = open_viewer(
                    *window_size, "yolo-pose — live",
                    headless_dir=args.save_dir,
                    resizable=args.resizable)
            except ViewerError as exc:
                print(f"no display available ({exc}); using headless output",
                      file=sys.stderr)
                args.no_window = True
        if args.no_window:
            from .render.viewer import HeadlessWriter
            if not args.save_dir:
                print("headless mode needs --save-dir", file=sys.stderr)
                return EXIT_ERROR
            viewer = HeadlessWriter(camera.format.width, camera.format.height,
                                    directory=args.save_dir,
                                    every=args.save_every)
            display = "headless-files"

        if args.record:
            recorder = _start_recorder(args.record, camera.format.width,
                                       camera.format.height, args.fps)

        parts = []
        if detector is not None:
            parts.append("姿态")
        if recognizer is not None:
            parts.append("人脸")
        tasks = " + ".join(parts)
        inference = detector.backend_name if detector is not None else (
            recognizer.detector.backend_name if recognizer else "none")
        size_note = ""
        if display_scale != 1.0:
            size_note = (f"  window={window_size[0]}x{window_size[1]} "
                         f"(x{display_scale:g})")
        lock_note = "" if args.resizable else "  (window locked)"
        print(f"camera {camera.device}: {camera.format}  capture={backend}  "
              f"inference={inference}  display={display}{size_note}{lock_note}")
        if args.detect_every > 1:
            print(f"detecting every {args.detect_every} frame(s); the overlay is "
                  f"reused in between")
        mode = ("async" if (args.async_infer is not False)
                else "parallel" if run_parallel else "sequential")
        print(f"tasks: {tasks}  ({mode})")
        print("press q or Esc in the window (or Ctrl-C here) to stop")

        frame_index = 0
        smoothed = 0.0
        with camera:
            # Asynchronous inference: the preview never waits for the model.
            # Frames go into a latest-only slot, the worker keeps the newest
            # overlay available, and the render loop draws at camera speed.
            use_async = (args.async_infer is not False) and (
                detector is not None or recognizer is not None)
            pose_worker = None
            face_worker = None
            if use_async:
                from .vision.async_infer import InferenceWorker
                if detector is not None:
                    pose_worker = InferenceWorker(
                        lambda f: detector.detect(f, keypoint_threshold=args.kpt_conf),
                        name="yolo-pose-pose")
                if recognizer is not None:
                    face_worker = InferenceWorker(
                        lambda f: recognizer.recognize(
                            f, score_threshold=args.face_score),
                        name="yolo-pose-face")
                # Prime one result so the very first drawn frame is annotated.
                if pose_worker is not None:
                    last_pose_result = pose_worker.wait_for_result(5.0)
                if face_worker is not None:
                    last_face_result = face_worker.wait_for_result(5.0)

            # Cached detection results: on skipped frames the overlay is redrawn
            # from the previous detections, which is both cheaper and simpler
            # than re-running inference.
            last_pose_result = None
            last_face_result = None
            detect_every = max(1, int(args.detect_every))

            while True:
                frame = camera.read()
                start_time = time.perf_counter()

                skipped = (detect_every > 1 and (frame_index % detect_every)
                           and (last_pose_result is not None
                                or last_face_result is not None))
                if skipped:
                    # Redraw the previous detections on the fresh frame: the
                    # display keeps the camera's frame rate while inference runs
                    # less often.
                    canvas = np.ascontiguousarray(frame.copy())
                    person_count = 0
                    if last_pose_result is not None:
                        canvas = _annotate(frame, last_pose_result, line_width=2.0)
                        person_count = len(last_pose_result.poses)
                    face_count = 0
                    if last_face_result is not None:
                        from .face.recognizer import draw_faces
                        draw_faces(canvas, last_face_result,
                                   threshold=args.threshold)
                        face_count = len(last_face_result.faces)
                    elapsed = time.perf_counter() - start_time
                    smoothed = (elapsed if smoothed == 0
                                else 0.9 * smoothed + 0.1 * elapsed)
                    hud = [f"FPS {1.0 / max(smoothed, 1e-6):5.1f}",
                           f"person {person_count}"]
                    if recognizer is not None:
                        hud.append(f"face {face_count}")
                    hud.append(f"{inference}/{backend} 1/{detect_every}")
                    draw_hud(canvas, hud)
                    if viewer is not None:
                        viewer.show(canvas)
                        if hasattr(viewer, "poll_events") and not viewer.poll_events():
                            break
                    if recorder is not None and recorder.stdin is not None:
                        try:
                            recorder.stdin.write(
                                np.ascontiguousarray(canvas).tobytes())
                            frames_written += 1
                        except (BrokenPipeError, OSError):
                            recorder = None
                    frame_index += 1
                    if args.frames and frame_index >= args.frames:
                        break
                    continue

                pose_serial = face_serial = 0
                if pose_worker is not None or face_worker is not None:
                    if pose_worker is not None:
                        pose_worker.submit(frame)
                    if face_worker is not None:
                        face_worker.submit(frame)
                    # Pick up any newer overlay produced since the last frame.
                    if pose_worker is not None:
                        result, pose_serial = pose_worker.latest()
                        if result is not None:
                            last_pose_result = result
                    if face_worker is not None:
                        fres, face_serial = face_worker.latest()
                        if fres is not None:
                            last_face_result = fres
                    canvas = np.ascontiguousarray(frame.copy())
                    person_count = 0
                    if last_pose_result is not None:
                        canvas = _annotate(frame, last_pose_result, line_width=2.0)
                        person_count = len(last_pose_result.poses)
                    face_count = 0
                    if last_face_result is not None:
                        from .face.recognizer import draw_faces
                        draw_faces(canvas, last_face_result,
                                   threshold=args.threshold)
                        face_count = len(last_face_result.faces)
                    latency = 0.0
                    if pose_worker is not None:
                        latency = pose_worker.stats.avg_latency
                    elif face_worker is not None:
                        latency = face_worker.stats.avg_latency
                    elapsed = time.perf_counter() - start_time
                    smoothed = (elapsed if smoothed == 0
                                else 0.9 * smoothed + 0.1 * elapsed)
                    hud = [f"FPS {1.0 / max(smoothed, 1e-6):5.1f}",
                           f"person {person_count}"]
                    if recognizer is not None:
                        hud.append(f"face {face_count}")
                    hud.append(f"{inference}/{backend} async {latency * 1000:.0f}ms")
                    draw_hud(canvas, hud)
                    if viewer is not None:
                        viewer.show(_scale_for_display(canvas, display_scale)
                                    if display_scale != 1.0 else canvas)
                        if hasattr(viewer, "poll_events") and not viewer.poll_events():
                            break
                    if recorder is not None and recorder.stdin is not None:
                        try:
                            recorder.stdin.write(
                                np.ascontiguousarray(canvas).tobytes())
                            frames_written += 1
                        except (BrokenPipeError, OSError):
                            recorder = None
                    frame_index += 1
                    if args.frames and frame_index >= args.frames:
                        break
                    continue

                pose_future = face_future = None
                if executor is not None:
                    pose_future = executor.submit(
                        detector.detect, frame, args.kpt_conf)
                    face_future = executor.submit(
                        recognizer.recognize, frame, args.face_score)

                # --- pose pass ---
                if detector is not None:
                    result = (pose_future.result() if pose_future is not None
                              else detector.detect(frame,
                                                   keypoint_threshold=args.kpt_conf))
                    canvas = _annotate(frame, result, line_width=2.0)
                    person_count = len(result.poses)
                else:
                    canvas = np.ascontiguousarray(frame.copy())
                    person_count = 0

                # --- face pass ---
                face_result = None
                face_count = 0
                if recognizer is not None:
                    try:
                        face_result = (face_future.result()
                                       if face_future is not None
                                       else recognizer.recognize(
                                           frame,
                                           score_threshold=args.face_score))
                        from .face.recognizer import draw_faces
                        draw_faces(canvas, face_result, threshold=args.threshold)
                        face_count = len(face_result.faces)
                    except Exception as exc:
                        if frame_index == 0:
                            print(f"warning: face recognition failed ({exc})",
                                  file=sys.stderr)

                if detector is not None:
                    last_pose_result = result
                if recognizer is not None and face_result is not None:
                    last_face_result = face_result

                elapsed = time.perf_counter() - start_time
                smoothed = elapsed if smoothed == 0 else 0.9 * smoothed + 0.1 * elapsed

                hud = [f"FPS {1.0 / max(smoothed, 1e-6):5.1f}"]
                if detector is not None:
                    hud.append(f"person {person_count}")
                if recognizer is not None:
                    hud.append(f"face {face_count}")
                hud.append(f"{inference}/{backend}")
                draw_hud(canvas, hud)

                if viewer is not None:
                    try:
                        # Backends that can scale during the blit take the window
                        # size directly; the X11 protocol viewer cannot, so it
                        # gets a pre-scaled array instead.
                        # XPutImage scaling is not honoured by every server
                        # (Xwayland clips instead of stretching), so the frame is
                        # enlarged here with nearest-neighbour replication. That
                        # costs ~2 ms for 2x, versus a bilinear resize at 55 ms.
                        viewer.show(_scale_for_display(canvas, display_scale)
                                    if display_scale != 1.0 else canvas)
                    except TypeError:
                        viewer.show(_scale_for_display(canvas, display_scale)
                                    if display_scale != 1.0 else canvas)
                    if hasattr(viewer, "poll_events") and not viewer.poll_events():
                        break
                if recorder is not None and recorder.stdin is not None:
                    try:
                        recorder.stdin.write(np.ascontiguousarray(canvas).tobytes())
                        frames_written += 1
                    except (BrokenPipeError, OSError):
                        recorder = None

                frame_index += 1
                if args.frames and frame_index >= args.frames:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        for worker in (locals().get("pose_worker"), locals().get("face_worker")):
            if worker is not None:
                worker.close()
        if executor is not None:
            executor.shutdown(wait=False)
        if viewer is not None:
            viewer.close()
        if recorder is not None and recorder.stdin is not None:
            try:
                recorder.stdin.close()
                recorder.wait(timeout=10)
            except Exception:
                pass

    if frames_written:
        print(f"recorded {frames_written} frames to {args.record}")
    return EXIT_OK


def command_face(args) -> int:
    """Detect (and optionally identify) faces in images."""
    from .face.align import align_face
    from .face.recognizer import (DEFAULT_THRESHOLD, FaceGallery, FaceRecognizer,
                                  draw_faces)

    if not args.input:
        print("error: at least one input image is required", file=sys.stderr)
        return EXIT_ERROR

    gallery = None
    if args.gallery:
        try:
            gallery = FaceGallery.load(args.gallery, threshold=args.threshold)
            print(f"gallery: {args.gallery} "
                  f"({len(gallery)} entries, {len(gallery.people)} people)")
        except FileNotFoundError:
            if args.enroll:
                gallery = FaceGallery(threshold=args.threshold)
            else:
                print(f"error: gallery {args.gallery} not found", file=sys.stderr)
                return EXIT_ERROR

    try:
        recognizer = FaceRecognizer(
            detector_model=args.det_model,
            embedder_model=args.emb_model,
            backend=args.backend,
            gallery=gallery,
            threshold=args.threshold,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_MODEL

    if args.verbose:
        print(recognizer.describe())

    single = len(args.input) == 1
    output = args.output
    if not single and output and not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    payload = {}
    total_faces = 0
    status = EXIT_OK

    for path in args.input:
        image = load_image(path)
        if args.enroll:
            if recognizer.embedder is None:
                print("error: --enroll needs a face embedding model; "
                      "run `yolo-pose models pull --face`", file=sys.stderr)
                return EXIT_NO_MODEL
            added = recognizer.enroll(image, args.enroll,
                                      score_threshold=args.face_score)
            print(f"{path}: enrolled {added} face(s) as {args.enroll!r}")
            if added == 0:
                status = EXIT_ERROR
            continue

        result = recognizer.recognize(image, score_threshold=args.face_score)
        total_faces += len(result.faces)

        names = [f.name for f in result.faces]
        known = [n for n in names if n != "unknown"]
        print(f"{path}: {len(result.faces)} face(s), {len(known)} recognised "
              f"in {result.inference_seconds * 1000:.0f} ms "
              f"({result.detector_backend})")
        for index, entry in enumerate(result.faces, 1):
            print(f"  face {index}: {entry.name} "
                  f"similarity={entry.similarity:.3f} det={entry.score:.3f} "
                  f"box=[{entry.box[0]:.0f},{entry.box[1]:.0f},"
                  f"{entry.box[2]:.0f},{entry.box[3]:.0f}]")

        if args.save_crops:
            for index, entry in enumerate(result.faces):
                crop = align_face(image, entry.face)
                name = entry.name if entry.name != "unknown" else "unknown"
                target = os.path.join(
                    args.save_crops,
                    f"{os.path.splitext(os.path.basename(path))[0]}_{index}_{name}.png")
                save_image(target, crop)

        if output is not None:
            canvas = np.ascontiguousarray(image.copy())
            draw_faces(canvas, result, threshold=args.threshold,
                       show_landmarks=not args.no_landmarks)
            if single:
                target = output
            else:
                target = os.path.join(output, os.path.basename(path))
                target = os.path.splitext(target)[0] + "_faces.png"
            save_image(target, canvas)
            print(f"  -> {target}")

        payload[path] = result.as_dict()

    if args.save_gallery:
        if gallery is None:
            gallery = recognizer.gallery
        gallery.save(args.save_gallery)
        print(f"saved gallery ({len(gallery)} entries) to {args.save_gallery}")

    if args.json_path:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.json_path == "-":
            print(text)
        else:
            with open(args.json_path, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
            print(f"wrote {args.json_path}")

    if not args.enroll and total_faces == 0:
        status = EXIT_ERROR
    return status


def command_demo(args) -> int:
    """Serve the local browser demo."""
    from .demo import serve

    try:
        return serve(host=args.host, port=args.port, pose_model=args.model,
                     face_detector=args.det_model, face_embedder=args.emb_model,
                     backend=args.backend, open_browser=args.open,
                     auto_port=not args.strict_port)
    except OSError as exc:
        print(f"error: cannot bind {args.host}:{args.port} ({exc})", file=sys.stderr)
        print("       try another port, e.g. --port 8899", file=sys.stderr)
        return EXIT_ERROR


def command_devices(args) -> int:
    """Report accelerator availability and what is blocking it."""
    from .core.accelerators import diagnose, format_diagnosis

    report = diagnose()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_diagnosis(report))
    return EXIT_OK


def command_models(args) -> int:
    if args.models_command == "list":
        rows = list_models()
        if args.json:
            print(json.dumps([
                {"name": n, "filename": f, "installed": i, "path": p}
                for n, f, i, p in rows], indent=2))
            return EXIT_OK
        print(f"{'name':<16} {'installed':<10} {'size':>7}  description")
        for name, spec in {**MODELS, **FACE_MODELS}.items():
            found = find_model(name)
            mark = "yes" if found else "no"
            print(f"{name:<16} {mark:<10} {spec.size_mb:6.1f}M  {spec.description}")
            if found:
                print(f"{'':<16} {'':<10} {'':>7}   {found}")
        print("\nlicense: model weights are AGPL-3.0 (Ultralytics); the code is MIT.")
        return EXIT_OK

    registry = FACE_MODELS if getattr(args, "face", False) else MODELS
    if getattr(args, "all", False):
        names = list(registry)
    else:
        names = [args.name]
        if args.name == DEFAULT_MODEL and getattr(args, "face", False):
            names = list(FACE_MODELS)
    for name in names:
        try:
            path = pull_model(name, destination=args.output,
                              force=args.force)
            print(path)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
    return EXIT_OK


def command_inspect(args) -> int:
    from .core.onnx_graph import describe, load_model
    from .core.runtime import SUPPORTED_OPS

    path = find_model(args.model)
    if path is None:
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return EXIT_NO_MODEL
    model = load_model(path)
    print(f"file           : {path}")
    print(f"size           : {os.path.getsize(path) / 1e6:.2f} MB")
    print(describe(model))

    missing = sorted(set(model.op_histogram()) - SUPPORTED_OPS)
    if missing:
        print(f"\nWARNING: unsupported operators: {', '.join(missing)}")
    else:
        print("\nall operators are supported by the built-in NumPy runtime")

    if args.ops:
        print("\nnodes:")
        for index, node in enumerate(model.nodes):
            print(f"  {index:4d} {node.op_type:<20} {node.name}")
    return EXIT_OK


def command_selftest(args) -> int:
    """Exercise parsing, inference, decoding and rendering on a synthetic frame."""
    from .core.onnx_graph import load_model
    from .core.runtime import NumpyEngine, SUPPORTED_OPS
    from .vision.imaging import letterbox, to_network_input

    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        mark = "PASS" if condition else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    print("yolo-pose selftest")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  numpy       : {np.__version__}")
    backends = available_backends()
    print(f"  backends    : {', '.join(k for k, v in backends.items() if v)}")

    path = find_model(args.model)
    check("model present", path is not None, path or "run `yolo-pose models pull`")
    if path is None:
        return EXIT_NO_MODEL

    model = load_model(path)
    check("onnx parsed", len(model.nodes) > 0,
          f"{len(model.nodes)} nodes, {len(model.initializers)} tensors")
    missing = sorted(set(model.op_histogram()) - SUPPORTED_OPS)
    check("operators supported", not missing, ", ".join(missing) or "all covered")

    engine = NumpyEngine(path)
    size = engine.input_shape
    check("static input shape", size is not None, str(size))

    height, width = (size[2], size[3]) if size else (640, 640)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    padded, info = letterbox(frame, (width, height))
    tensor = to_network_input(padded)
    output = engine.run({engine.input_name: tensor})
    key = next(iter(output))
    array = output[key]
    check("forward pass", array.ndim == 3 and np.isfinite(array).all(),
          f"{key} {array.shape}")

    detector = Detector(path, backend="numpy", verbose=False)
    result = detector.detect(frame)
    check("decode pipeline", isinstance(result.poses, list),
          f"{len(result.poses)} pose(s) on a synthetic frame")

    canvas = _annotate(frame, result)
    check("render pipeline", canvas.shape == frame.shape, str(canvas.shape))

    try:
        from .vision.camera import list_cameras
        devices = list_cameras()
        print(f"  [info] capture devices: {', '.join(devices) or 'none'}")
    except Exception as exc:
        print(f"  [info] camera probe failed: {exc}")

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return EXIT_ERROR
    print("\nall checks passed")
    return EXIT_OK


COMMANDS = {
    "image": command_image,
    "devices": command_devices,
    "face": command_face,
    "demo": command_demo,
    "camera": command_camera,
    "joints": command_joints,
    "models": command_models,
    "inspect": command_inspect,
    "selftest": command_selftest,
}


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse guards this
        parser.error(f"unknown command {args.command}")
        return EXIT_ERROR
    try:
        return handler(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_MODEL
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(args, "verbose", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
