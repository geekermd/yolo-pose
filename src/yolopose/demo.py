"""A tiny local web demo for yolo-pose.

Serves a single page that lets you drop in an image (or load the bundled sample)
and see pose estimation and face recognition results rendered live.  Implemented
on :mod:`http.server` so it needs no web framework, and it binds to 127.0.0.1 by
default so nothing is exposed to the network.

    yolo-pose demo                # then open http://127.0.0.1:8765
    yolo-pose demo --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .models import find_face_model, find_model
from .render.skeleton import draw_angle_labels, draw_pose
from .vision.detector import Detector
from .vision.imaging import load_image, save_image
from .vision.pose import KEYPOINT_NAMES

MAX_UPLOAD = 24 * 1024 * 1024          # 24 MB is plenty for a photo


def png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB ``uint8`` array as PNG bytes, entirely in memory."""
    import struct
    import zlib

    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    array = array[:, :, :3]

    height, width = array.shape[:2]
    raw = bytearray()
    stride = width * 3
    for row in range(height):
        raw.append(0)                      # filter type 0 (None)
        raw.extend(array[row].tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


def _png_base64(array: np.ndarray) -> str:
    """Encode an RGB array as a base64 PNG data URL."""
    return "data:image/png;base64," + base64.b64encode(png_bytes(array)).decode("ascii")


class DemoState:
    """Lazily-built detector/recognizer shared across requests."""

    def __init__(self, pose_model: Optional[str], face_detector: Optional[str],
                 face_embedder: Optional[str], backend: str = "auto"):
        self._lock = threading.Lock()
        self._pose: Optional[Detector] = None
        self._face = None
        self.pose_model = pose_model or find_model("yolov8n-pose")
        self.face_detector = face_detector or find_face_model("detector")
        self.face_embedder = face_embedder or find_face_model("embedder")
        self.backend = backend
        self.gallery_path = _find_demo_gallery()
        self.stats = {"requests": 0, "pose": 0, "face": 0}

    @property
    def pose(self) -> Detector:
        with self._lock:
            if self._pose is None:
                if not self.pose_model:
                    raise RuntimeError(
                        "no pose model found — run `yolo-pose models pull`")
                self._pose = Detector(self.pose_model, backend=self.backend)
            return self._pose

    @property
    def face(self):
        with self._lock:
            if self._face is None:
                from .face import FaceGallery, FaceRecognizer

                gallery = None
                if self.gallery_path:
                    try:
                        gallery = FaceGallery.load(self.gallery_path)
                    except Exception:
                        gallery = None
                self._face = FaceRecognizer(
                    detector_model=self.face_detector,
                    embedder_model=self.face_embedder,
                    backend=self.backend, gallery=gallery)
            return self._face

    def capabilities(self) -> Dict[str, Any]:
        return {
            "pose_model": os.path.basename(self.pose_model) if self.pose_model else None,
            "face_detector": os.path.basename(self.face_detector) if self.face_detector else None,
            "face_embedder": os.path.basename(self.face_embedder) if self.face_embedder else None,
            "pose_ready": self._pose is not None,
            "face_ready": self._face is not None,
            "backend": self._pose.backend_name if self._pose else self.backend,
            "gallery": self.gallery_path,
        }


def _find_demo_gallery() -> Optional[str]:
    """Locate a gallery to preload so the demo shows real names."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for candidate in (
        os.path.join(here, "examples", "demo_gallery.npz"),
        os.path.join(os.getcwd(), "examples", "demo_gallery.npz"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _decode_upload(payload: Dict[str, Any]) -> np.ndarray:
    """Decode the JSON request body into an RGB image array."""
    import tempfile

    encoded = payload.get("image", "")
    if not encoded:
        raise ValueError("request has no 'image' field")
    if "," in encoded and encoded.strip().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=False)
    except Exception as exc:
        raise ValueError(f"image is not valid base64: {exc}") from exc
    if len(raw) > MAX_UPLOAD:
        raise ValueError("image is too large (limit 24 MB)")
    if len(raw) < 32:
        raise ValueError("image data is empty")

    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as temporary:
        temporary.write(raw)
        path = temporary.name
    try:
        return load_image(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _handle_analyze(state: DemoState, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run both pipelines over an uploaded image and return annotated PNGs."""
    task = payload.get("task", "both")
    image = _decode_upload(payload)
    state.stats["requests"] += 1
    height, width = image.shape[:2]

    response: Dict[str, Any] = {
        "width": width,
        "height": height,
        "original": _png_base64(image),
        "warnings": [],
    }

    if task in ("pose", "both"):
        start = time.perf_counter()
        result = state.pose.detect(image)
        elapsed = time.perf_counter() - start
        state.stats["pose"] += 1

        canvas = np.ascontiguousarray(image.copy())
        for pose, joints in zip(result.poses, result.joints):
            draw_pose(canvas, pose, show_box=True, show_score=True)
            draw_angle_labels(canvas, pose, joints, scale=1)

        persons = []
        for pose, joints in zip(result.poses, result.joints):
            persons.append({
                "score": round(float(pose.score), 4),
                "box": [round(float(v), 1) for v in pose.box],
                "visible": int(np.count_nonzero(
                    pose.keypoint_scores >= state.pose.conf_threshold)),
                "angles": {k: round(float(v), 1)
                           for k, v in sorted(joints.angles.items())},
                "keypoints": {
                    name: [
                        round(float(pose.keypoints[i, 0]), 1),
                        round(float(pose.keypoints[i, 1]), 1),
                        round(float(pose.keypoint_scores[i]), 3),
                    ]
                    for i, name in enumerate(KEYPOINT_NAMES)
                },
            })
        response["pose"] = {
            "image": _png_base64(canvas),
            "count": len(result.poses),
            "inference_ms": round(elapsed * 1000, 1),
            "backend": result.backend,
            "persons": persons,
        }

    if task in ("face", "both"):
        try:
            face = state.face
        except Exception as exc:
            response["warnings"].append(f"face pipeline unavailable: {exc}")
            face = None

        if face is not None:
            from .face.recognizer import draw_faces

            start = time.perf_counter()
            result = face.recognize(image)
            elapsed = time.perf_counter() - start
            state.stats["face"] += 1

            canvas = np.ascontiguousarray(image.copy())
            draw_faces(canvas, result)
            response["face"] = {
                "image": _png_base64(canvas),
                "count": len(result.faces),
                "inference_ms": round(elapsed * 1000, 1),
                "backend": result.detector_backend,
                "faces": [
                    {
                        "name": entry.name,
                        "similarity": round(float(entry.similarity), 4),
                        "det_score": round(float(entry.score), 4),
                        "box": [round(float(v), 1) for v in entry.box],
                        "landmarks": [[round(float(x), 1), round(float(y), 1)]
                                      for x, y in entry.face.landmarks],
                    }
                    for entry in result.faces
                ],
                "gallery_size": len(face.gallery),
            }

    return response


def _page(state: DemoState, sample_path: Optional[str]) -> str:
    """Build the single-page UI."""
    capabilities = state.capabilities()
    title = "yolo-pose 本地演示"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #171a23; --border: #262b38;
    --text: #e6e8ee; --muted: #98a0b3; --accent: #4da3ff; --ok: #3ddc84;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.55 system-ui, "Noto Sans CJK SC", sans-serif; }}
  header {{ padding: 18px 24px; border-bottom: 1px solid var(--border);
            display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
  h1 {{ font-size: 17px; margin: 0; font-weight: 650; }}
  .muted {{ color: var(--muted); font-size: 12.5px; }}
  main {{ padding: 20px 24px 60px; max-width: 1400px; margin: 0 auto; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border);
            border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
  .row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
  button {{ background: #223047; color: var(--text); border: 1px solid #33475f;
            border-radius: 7px; padding: 8px 14px; cursor: pointer; font-size: 13.5px; }}
  button:hover {{ background: #2b3d5c; }}
  button.primary {{ background: var(--accent); border-color: var(--accent); color: #06121f;
                    font-weight: 600; }}
  button.primary:hover {{ filter: brightness(1.1); }}
  button:disabled {{ opacity: .45; cursor: not-allowed; }}
  select, input[type=range] {{ background: #1d2330; color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px 9px; }}
  #drop {{ border: 2px dashed var(--border); border-radius: 10px; padding: 32px;
           text-align: center; color: var(--muted); transition: .15s; }}
  #drop.hot {{ border-color: var(--accent); color: var(--text);
               background: rgba(77,163,255,.06); }}
  .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }}
  .card {{ background: var(--panel); border: 1px solid var(--border);
           border-radius: 10px; overflow: hidden; }}
  .card h2 {{ font-size: 13px; margin: 0; padding: 11px 14px;
              border-bottom: 1px solid var(--border); color: var(--muted);
              font-weight: 600; letter-spacing: .3px; text-transform: uppercase; }}
  .card img {{ display: block; width: 100%; height: auto; background: #0a0c11; }}
  .meta {{ padding: 11px 14px; font-size: 12.5px; color: var(--muted);
           border-top: 1px solid var(--border); }}
  .meta b {{ color: var(--text); font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
  code {{ background: #0d1017; padding: 2px 6px; border-radius: 4px;
          font-size: 12px; color: #a8d5ff; }}
  .pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px;
           font-size: 11.5px; border: 1px solid var(--border); color: var(--muted); }}
  .pill.ok {{ border-color: rgba(61,220,132,.5); color: var(--ok); }}
  .pill.no {{ border-color: rgba(255,120,120,.45); color: #ff9a9a; }}
  #status {{ min-height: 20px; font-size: 13px; }}
  .spin {{ display: inline-block; width: 12px; height: 12px; margin-right: 7px;
           border: 2px solid var(--border); border-top-color: var(--accent);
           border-radius: 50%; animation: sp .7s linear infinite; }}
  @keyframes sp {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<header>
  <h1>yolo-pose 本地演示</h1>
  <span class="muted">纯 NumPy ONNX 推理 &middot; 姿态识别 + 人脸识别</span>
  <span style="flex:1"></span>
  <span class="pill {'ok' if capabilities['pose_model'] else 'no'}">
    姿态模型 {capabilities['pose_model'] or '缺失'}</span>
  <span class="pill {'ok' if capabilities['face_detector'] else 'no'}">
    人脸模型 {capabilities['face_detector'] or '缺失'}</span>
</header>

<main>
  <div class="panel">
    <div class="row" style="margin-bottom:12px">
      <label class="muted">任务
        <select id="task">
          <option value="both">姿态 + 人脸</option>
          <option value="pose">仅姿态</option>
          <option value="face">仅人脸</option>
        </select>
      </label>
      <label class="muted">后端
        <select id="backend">
          <option value="auto">自动（优先 onnxruntime）</option>
          <option value="numpy">纯 NumPy</option>
          <option value="onnxruntime">ONNX Runtime</option>
        </select>
      </label>
      <button class="primary" id="run" disabled>开始识别</button>
      <button id="sample">载入样例图</button>
      <span class="muted">或直接把图片拖到下面 / Ctrl+V 粘贴</span>
    </div>
    <div id="drop">
      拖放图片到这里，或点击选择文件
      <input type="file" id="file" accept="image/*" style="display:none">
    </div>
    <div id="status" style="margin-top:12px"></div>
  </div>

  <div class="grid" id="results"></div>
</main>

<script>
const state = {{ image: null, name: '', busy: false }};
const $ = id => document.getElementById(id);

function setStatus(html, spinning) {{
  $('status').innerHTML = (spinning ? '<span class="spin"></span>' : '') + html;
}}

function arm() {{
  $('run').disabled = !state.image || state.busy;
}}

async function fileToDataURL(file) {{
  return new Promise((resolve, reject) => {{
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  }});
}}

async function acceptFile(file) {{
  if (!file || !file.type.startsWith('image/')) {{
    setStatus('<span style="color:#ff9a9a">请选择图片文件</span>'); return;
  }}
  state.image = await fileToDataURL(file);
  state.name = file.name || 'image';
  $('drop').innerHTML = '<img src="' + state.image +
      '" style="max-height:180px;border-radius:6px"><div class="muted" style="margin-top:8px">' +
      state.name + '</div><input type="file" id="file" accept="image/*" style="display:none">';
  bindDrop();
  arm();
  setStatus('已载入 <b>' + state.name + '</b>，点击「开始识别」');
}}

function bindDrop() {{
  const drop = $('drop');
  drop.onclick = () => $('file').click();
  $('file').onchange = e => e.target.files[0] && acceptFile(e.target.files[0]);
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {{
    e.preventDefault(); drop.classList.add('hot');
  }}));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {{
    e.preventDefault(); drop.classList.remove('hot');
  }}));
  drop.addEventListener('drop', e => e.dataTransfer.files[0] &&
      acceptFile(e.dataTransfer.files[0]));
}}

document.addEventListener('paste', e => {{
  for (const item of (e.clipboardData?.items || [])) {{
    if (item.type.startsWith('image/')) {{ acceptFile(item.getAsFile()); break; }}
  }}
}});

$('sample').onclick = async () => {{
  setStatus('正在载入样例图…', true);
  const response = await fetch('/api/sample');
  const payload = await response.json();
  if (payload.error) {{ setStatus('<span style="color:#ff9a9a">' + payload.error + '</span>'); return; }}
  state.image = payload.image; state.name = payload.name;
  $('drop').innerHTML = '<img src="' + state.image +
      '" style="max-height:180px;border-radius:6px"><div class="muted" style="margin-top:8px">' +
      state.name + '（内置样例）</div><input type="file" id="file" accept="image/*" style="display:none">';
  bindDrop(); arm();
  setStatus('已载入样例图，点击「开始识别」');
}};

$('run').onclick = async () => {{
  if (!state.image || state.busy) return;
  state.busy = true; arm();
  setStatus('正在推理…（纯 NumPy 约 0.15 秒/帧，首次加载模型稍慢）', true);
  const started = performance.now();
  try {{
    const response = await fetch('/api/analyze', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{image: state.image, task: $('task').value}})
    }});
    const payload = await response.json();
    if (payload.error) throw new Error(payload.error);
    render(payload);
    const total = ((performance.now() - started) / 1000).toFixed(2);
    setStatus('完成，总耗时 <b>' + total + ' s</b>');
  }} catch (err) {{
    setStatus('<span style="color:#ff9a9a">失败：' + err.message + '</span>');
  }} finally {{
    state.busy = false; arm();
  }}
}};

function render(payload) {{
  const cards = [];
  cards.push(card('原图', payload.original,
    payload.width + ' × ' + payload.height + ' px'));
  if (payload.pose) {{
    const persons = payload.pose.persons.map((p, i) => {{
      const angles = Object.entries(p.angles)
        .map(([k, v]) => k + ' ' + v + '°').join('、') || '（关键点不足）';
      return '<tr><td>#' + (i + 1) + '</td><td>' + (p.score * 100).toFixed(1) + '%</td>' +
             '<td>' + p.visible + '/17</td><td>' + angles + '</td></tr>';
    }}).join('');
    const table = payload.pose.count
      ? '<table><tr><th>#</th><th>置信度</th><th>可见关键点</th><th>关节角度</th></tr>' +
        persons + '</table>'
      : '<div class="muted">未检测到人体</div>';
    cards.push(card('人体姿态（COCO-17 骨架 + 关节角度）', payload.pose.image,
      '检出 <b>' + payload.pose.count + '</b> 人 &middot; 推理 <b>' +
      payload.pose.inference_ms + ' ms</b> &middot; 后端 <code>' +
      payload.pose.backend + '</code>' + table));
  }}
  if (payload.face) {{
    const rows = payload.face.faces.map((f, i) =>
      '<tr><td>#' + (i + 1) + '</td><td>' + f.name + '</td>' +
      '<td>' + f.similarity.toFixed(3) + '</td>' +
      '<td>' + f.det_score.toFixed(3) + '</td></tr>').join('');
    const table = payload.face.count
      ? '<table><tr><th>#</th><th>身份</th><th>相似度</th><th>检测分</th></tr>' +
        rows + '</table>'
      : '<div class="muted">未检测到人脸</div>';
    cards.push(card('人脸检测与识别（SCRFD + 512 维特征）', payload.face.image,
      '检出 <b>' + payload.face.count + '</b> 张 &middot; 推理 <b>' +
      payload.face.inference_ms + ' ms</b> &middot; 底库 <b>' +
      payload.face.gallery_size + '</b> 条' + table));
  }}
  for (const warning of (payload.warnings || [])) {{
    cards.push('<div class="panel muted">⚠ ' + warning + '</div>');
  }}
  $('results').innerHTML = cards.join('');
}}

function card(title, image, meta) {{
  return '<div class="card"><h2>' + title + '</h2>' +
         '<img src="' + image + '" alt="' + title + '">' +
         '<div class="meta">' + meta + '</div></div>';
}}

bindDrop();
</script>
</body>
</html>
"""


def _sample_payload() -> Dict[str, Any]:
    """Return the bundled example image, if present."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    candidates = [
        os.path.join(here, "examples", "sample_input.jpg"),
        os.path.join(os.getcwd(), "examples", "sample_input.jpg"),
        "/tmp/test_zidane.jpg",
    ]
    for path in candidates:
        if os.path.exists(path):
            return {"name": os.path.basename(path),
                    "image": _png_base64(load_image(path))}
    return {"error": "样例图不可用：未找到 examples/sample_input.jpg"}


def make_handler(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "yolo-pose-demo/1.0"

        def log_message(self, fmt, *args):   # keep the console readable
            if state.stats["requests"] % 1 == 0:
                pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            self._send(json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8", status)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(_page(state, None).encode("utf-8"),
                           "text/html; charset=utf-8")
            elif self.path == "/api/capabilities":
                self._send_json(state.capabilities())
            elif self.path == "/api/sample":
                self._send_json(_sample_payload())
            elif self.path == "/favicon.ico":
                self._send(b"", "image/x-icon")
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/analyze":
                self._send_json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_UPLOAD * 2:
                self._send_json({"error": "invalid request size"}, 400)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as exc:
                self._send_json({"error": f"malformed JSON: {exc}"}, 400)
                return
            try:
                self._send_json(_handle_analyze(state, payload))
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)

    return Handler


def _probe_existing(host: str, port: int, timeout: float = 1.5) -> bool:
    """True when something already answers like a yolo-pose demo on this port."""
    import urllib.request

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    try:
        with urllib.request.urlopen(
                f"http://{probe_host}:{port}/api/capabilities", timeout=timeout) as reply:
            return reply.status == 200
    except Exception:
        return False


def serve(host: str = "127.0.0.1", port: int = 8765,
          pose_model: Optional[str] = None, face_detector: Optional[str] = None,
          face_embedder: Optional[str] = None, backend: str = "auto",
          open_browser: bool = False, auto_port: bool = True) -> int:
    """Start the demo server and block until interrupted.

    If the requested port is already taken, an already-running yolo-pose demo is
    reused (and its page opened); otherwise the next free port is chosen.  That
    keeps a desktop shortcut from failing with "Address already in use" just
    because an earlier instance is still up.
    """
    state = DemoState(pose_model, face_detector, face_embedder, backend)
    capabilities = state.capabilities()

    if auto_port and _probe_existing(host, port):
        url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/"
        print(f"yolo-pose demo is already running at {url}")
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        return 0

    server = None
    chosen = port
    attempts = [port] + ([port + offset for offset in range(1, 20)] if auto_port else [])
    last_error: Optional[OSError] = None
    for candidate in attempts:
        try:
            server = ThreadingHTTPServer((host, candidate), make_handler(state))
            chosen = candidate
            break
        except OSError as exc:
            last_error = exc
            continue
    if server is None:
        print(f"error: no free port in {attempts[0]}..{attempts[-1]} ({last_error})",
              file=sys.stderr)
        return 1
    if chosen != port:
        print(f"port {port} is busy; using {chosen} instead")

    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{chosen}/"

    print("yolo-pose 本地演示")
    print(f"  姿态模型 : {capabilities['pose_model'] or '未找到'}")
    print(f"  人脸模型 : {capabilities['face_detector'] or '未找到'}"
          f"{' + ' + capabilities['face_embedder'] if capabilities['face_embedder'] else ''}")
    print(f"  监听地址 : {url}")
    if host not in ("127.0.0.1", "localhost"):
        print("  注意     : 正在监听非本地地址，同网段主机可访问")
    print("  停止服务 : Ctrl-C")
    print()

    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
    return 0
