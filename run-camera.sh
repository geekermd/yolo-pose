#!/usr/bin/env bash
#
# One-click launcher for live camera pose + face recognition.
#
#   ./run-camera.sh                 # live window, pose + face recognition
#   ./run-camera.sh --pose-only     # skip the face pipeline (lighter)
#   ./run-camera.sh --save          # also write annotated frames to ./frames
#   ./run-camera.sh --device 1      # pick another camera
#   ./run-camera.sh --list          # show capture devices and exit
#
# It works both from a source checkout (adds src/ to PYTHONPATH) and from an
# installed copy (uses the `yolo-pose` command).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GALLERY="$HERE/examples/demo_gallery.npz"
DEVICE="/dev/video0"
EXTRA=()
POSE_ONLY=0
FACE_ONLY=0
SAVE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pose-only) POSE_ONLY=1; FACE_ONLY=0; shift ;;
        --face-only) FACE_ONLY=1; POSE_ONLY=0; shift ;;
        --save) SAVE=1; shift ;;
        --list)
            PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}" \
                python3 -m yolopose camera --list
            exit $?
            ;;
        --device) DEVICE="$2"; shift 2 ;;
        --gallery) GALLERY="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------- environment
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is not installed" >&2
    exit 1
fi

# Always prefer the sources in this directory: a system-wide install may be an
# older release that lacks newer options, and that mismatch is confusing.
if [[ -d "$HERE/src/yolopose" && -z "${FORCE_INSTALLED:-}" ]]; then
    RUNNER=(python3 -m yolopose)
    export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
elif command -v yolo-pose >/dev/null 2>&1; then
    RUNNER=(yolo-pose)
else
    echo "error: neither $HERE/src/yolopose nor a 'yolo-pose' command was found" >&2
    exit 1
fi

# NumPy is the only hard requirement.
if ! "${RUNNER[@]}" --version >/dev/null 2>&1; then
    echo "yolo-pose could not start. Check the installation with:" >&2
    echo "  python3 -c 'import numpy; print(numpy.__version__)'" >&2
    exit 1
fi

# ---------------------------------------------------------------- model check
MODELS_DIR="$HERE/models"
missing=0
for model in yolov8n-pose.onnx; do
    if [[ ! -f "$MODELS_DIR/$model" ]] \
       && [[ ! -f "/usr/share/yolo-pose/models/$model" ]]; then
        echo "missing pose model: $model" >&2
        missing=1
    fi
done

if [[ "$POSE_ONLY" -eq 0 ]]; then
    for model in scrfd_500m.onnx w600k_mbf.onnx; do
        if [[ ! -f "$MODELS_DIR/$model" ]] \
           && [[ ! -f "$HOME/.local/share/yolo-pose/models/$model" ]]; then
            echo "missing face model: $model  (run: yolo-pose models pull --face)" >&2
            echo "falling back to pose-only mode" >&2
            POSE_ONLY=1
            break
        fi
    done
fi

if [[ "$missing" -eq 1 ]]; then
    echo "run:  yolo-pose models pull" >&2
    exit 2
fi

if [[ ! -e "$DEVICE" ]]; then
    echo "warning: $DEVICE does not exist; available devices:" >&2
    ls /dev/video* 2>/dev/null >&2 || echo "  (none)" >&2
fi

# ---------------------------------------------------------------- run
CMD=("${RUNNER[@]}" camera --device "$DEVICE" --capture-backend auto)

if [[ "$POSE_ONLY" -eq 1 ]]; then
    CMD+=(--pose)
fi

if [[ "$POSE_ONLY" -eq 0 ]]; then
    CMD+=(--faces)
    if [[ -f "$GALLERY" ]]; then
        CMD+=(--gallery "$GALLERY")
        echo "face gallery: $GALLERY"
    else
        echo "no gallery found at $GALLERY (faces will be labelled 'unknown')"
    fi
fi

if [[ "$FACE_ONLY" -eq 1 ]]; then
    # Skip the pose pass entirely rather than just raising its threshold.
    CMD+=(--no-pose)
fi

if [[ "$SAVE" -eq 1 ]]; then
    mkdir -p "$HERE/frames"
    CMD+=(--record "$HERE/frames/session.mp4")
    echo "recording annotated frames to $HERE/frames/session.mp4"
fi

echo "starting live recognition — press q or Esc in the window, or Ctrl-C here"
echo

exec "${CMD[@]}" "${EXTRA[@]+"${EXTRA[@]}"}"
