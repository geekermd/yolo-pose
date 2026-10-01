"""Camera capture backends.

Two implementations are provided:

* :class:`~yolopose.vision.camera.V4L2Camera` — pure-Python V4L2 with mmap
  streaming.  Preferred, because it needs no external process.
* :class:`FfmpegCamera` — drives ``ffmpeg`` as a subprocess.  Used automatically
  when the native path cannot negotiate a format, which happens in sandboxes that
  filter ``VIDIOC_G_FMT``/``VIDIOC_S_FMT``.

:func:`open_any_camera` picks the first backend that works and reports which one
was chosen.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .camera import (CameraError, CameraFormat, V4L2Camera, capture_devices,
                     open_camera as open_v4l2_camera)

__all__ = ["FfmpegCamera", "open_any_camera", "camera_backends", "CameraFormat",
           "backend_name"]


@dataclass
class _Probe:
    width: int
    height: int
    pixel_format: str


def _ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


class FfmpegCamera:
    """Capture raw frames by reading ``ffmpeg``'s stdout.

    This is the fallback for environments where the V4L2 ioctls needed to
    negotiate a format are unavailable.  ffmpeg performs the device setup itself,
    so the format is fixed at construction time.
    """

    def __init__(self, device="0", width: int = 640, height: int = 480,
                 fps: int = 30, pixel_format: str = "auto",
                 ffmpeg: Optional[str] = None, input_format: str = "v4l2"):
        if isinstance(device, int):
            device = f"/dev/video{device}"
        elif isinstance(device, str) and device.isdigit():
            device = f"/dev/video{device}"
        self.device = device
        self.requested = (pixel_format, width, height, fps)
        self.ffmpeg = ffmpeg or _ffmpeg_path()
        if not self.ffmpeg:
            raise CameraError("ffmpeg is not installed")

        self._input_format = input_format
        self._process: Optional[subprocess.Popen] = None
        self._closed = False

        probe = self._probe(width, height, fps, pixel_format)
        self.format = CameraFormat(fourcc=probe.pixel_format,
                                   width=probe.width, height=probe.height,
                                   size_image=probe.width * probe.height * 3)
        self._frame_bytes = probe.width * probe.height * 3
        self._start(width, height, fps, pixel_format)

    # -- probing ------------------------------------------------------------ #
    def _probe(self, width: int, height: int, fps: int,
               pixel_format: str) -> _Probe:
        """Decode one frame with ffmpeg to learn the real output geometry."""
        command = ([self.ffmpeg, "-hide_banner", "-loglevel", "info",
                    "-f", self._input_format, "-i", self.device,
                    "-frames:v", "30", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
                   if pixel_format.lower() in ("auto", "")
                   else [self.ffmpeg, "-hide_banner", "-loglevel", "info",
                         "-f", self._input_format, "-pixel_format",
                         _ffmpeg_pixel_format(pixel_format), "-i", self.device,
                         "-frames:v", "30", "-f", "rawvideo", "-pix_fmt",
                         "rgb24", "-"])
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise CameraError(f"ffmpeg probe timed out for {self.device}") from exc
        text = result.stderr.decode("utf-8", "replace")
        if result.returncode != 0:
            tail = text.strip().splitlines()[-3:]
            raise CameraError(
                f"ffmpeg could not open {self.device}: " + " | ".join(tail))

        match = re.search(r"Stream #0:\d+.*?,\s*(\d+)x(\d+)", text)
        if not match:
            raise CameraError("could not parse the capture size from ffmpeg output")
        probe_w, probe_h = int(match.group(1)), int(match.group(2))

        pixel = "RGB24"
        pixel_match = re.search(r"Video: (\w+)", text)
        if pixel_match:
            pixel = pixel_match.group(1).upper()
        return _Probe(width=probe_w, height=probe_h, pixel_format=pixel)

    # -- streaming ---------------------------------------------------------- #
    def _start(self, width: int, height: int, fps: int, pixel_format: str) -> None:
        input_args = ["-f", self._input_format]
        if pixel_format.lower() not in ("auto", ""):
            input_args += ["-pixel_format", _ffmpeg_pixel_format(pixel_format)]
        command = [self.ffmpeg, "-hide_banner", "-loglevel", "error"] + input_args + [
            "-i", self.device,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 2)
        # Wait for the first frame so read() does not return partial data.
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise CameraError(
                    f"ffmpeg exited early with code {self._process.returncode}")
            if self._has_data():
                return
            time.sleep(0.05)
        raise CameraError("timed out waiting for ffmpeg to produce a frame")

    def _has_data(self, timeout: float = 0.05) -> bool:
        import select

        assert self._process is not None and self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], timeout)
        return bool(ready)

    def start(self) -> None:
        return

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def read(self, raw: bool = False) -> np.ndarray:
        """Read one RGB frame from the ffmpeg pipe."""
        if self._closed or self._process is None or self._process.stdout is None:
            raise CameraError("camera is closed")
        if self._process.poll() is not None:
            raise CameraError(
                f"ffmpeg exited with code {self._process.returncode}")
        payload = self._read_exactly(self._frame_bytes)
        array = np.frombuffer(payload, dtype=np.uint8)
        return array.reshape(self.format.height, self.format.width, 3)

    def _read_exactly(self, count: int) -> bytes:
        assert self._process is not None and self._process.stdout is not None
        chunks = []
        remaining = count
        while remaining > 0:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                raise CameraError("ffmpeg stream ended unexpectedly")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def capabilities(self) -> Dict[str, object]:
        return {"driver": "ffmpeg", "card": self.device,
                "capture": True, "streaming": True, "readwrite": True}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        if self._process is not None and self._process.stdout is not None:
            try:
                self._process.stdout.close()
            except Exception:
                pass

    def __enter__(self) -> "FfmpegCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self):
        while not self._closed:
            yield self.read()


_FFMPEG_PIXEL_FORMATS = {
    "YUYV": "yuyv422", "YUY2": "yuyv422", "UYVY": "uyvy422",
    "MJPG": "mjpeg", "JPEG": "mjpeg",
    "RGB3": "rgb24", "RGB24": "rgb24", "BGR3": "bgr24", "BGR24": "bgr24",
    "GREY": "gray", "NV12": "nv12",
}


def _ffmpeg_pixel_format(name: str) -> str:
    return _FFMPEG_PIXEL_FORMATS.get(name.upper(), name.lower())


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #

def camera_backends() -> List[str]:
    """Backends available on this system, most preferred first."""
    backends = ["v4l2"]
    if _ffmpeg_path():
        backends.append("ffmpeg")
    return backends


def open_any_camera(device="0", width: int = 640, height: int = 480,
                    fps: int = 30, pixel_format: str = "auto",
                    backend: str = "auto", timeout: float = 2.0):
    """Open a camera with the best available backend.

    ``backend`` may be ``auto``, ``v4l2`` or ``ffmpeg``.  Returns the camera
    object; call :func:`backend_name` to see which one was used.
    """
    wanted = backend.lower()
    if wanted not in ("auto", "v4l2", "ffmpeg"):
        raise ValueError(f"unknown backend {backend!r}")

    errors: List[str] = []
    if wanted in ("auto", "v4l2"):
        try:
            camera = open_v4l2_camera(device, width, height, fps, pixel_format,
                                      timeout=timeout)
            setattr(camera, "_backend_name", "v4l2")
            return camera
        except CameraError as exc:
            errors.append(f"v4l2: {exc}")
            if wanted == "v4l2":
                raise

    if wanted in ("auto", "ffmpeg"):
        try:
            camera = FfmpegCamera(device, width, height, fps, pixel_format)
            setattr(camera, "_backend_name", "ffmpeg")
            return camera
        except CameraError as exc:
            errors.append(f"ffmpeg: {exc}")

    raise CameraError(
        "no camera backend could open " f"{device}: " + "; ".join(errors))


def backend_name(camera) -> str:
    """Name of the backend behind a camera object."""
    explicit = getattr(camera, "_backend_name", None)
    if explicit:
        return explicit
    return "ffmpeg" if isinstance(camera, FfmpegCamera) else "v4l2"
