#!/usr/bin/env bash
#
# Build a .deb for yolo-pose.
#
#   ./debian/build-deb.sh                     # architecture-independent (all)
#   ./debian/build-deb.sh --arch amd64        # native x86-64 package
#   ./debian/build-deb.sh --arch arm64        # native aarch64 package
#   ./debian/build-deb.sh --arch all --out dist
#
# The package is pure Python, so the only architecture-specific part is the
# `Architecture:` field and, for a native build, the interpreter the wrapper
# script points at.  `--arch arm64` therefore cross-builds a *correct* package
# as long as it is installed on an arm64 machine (the payload contains no
# binaries).  Pass --chroot to actually build inside an arm64 chroot instead.
#
# This script uses dpkg-deb directly, so it needs no debhelper and works offline.

set -euo pipefail
umask 022          # keep packaged files world-readable (0644/0755)

PACKAGE="yolo-pose"
VERSION="1.0.0"
ARCH="all"
OUT_DIR=""
CHROOT=""
KEEP_BUILD=0
INCLUDE_MODELS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'TXT'

options:
  --arch ARCH     all (default) | amd64 | arm64 | armhf | riscv64 | native
  --out DIR       output directory (default: <repo>/dist)
  --chroot        build inside a qemu/binfmt chroot for --arch (needs sudo)
  --no-models     do not bundle models/yolov8n-pose.onnx
  --keep          keep the staging directory
  -h, --help      show this help
TXT
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="${2:?--arch needs a value}"; shift 2 ;;
        --out) OUT_DIR="${2:?--out needs a value}"; shift 2 ;;
        --chroot) CHROOT=1; shift ;;
        --no-models) INCLUDE_MODELS=0; shift ;;
        --keep) KEEP_BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$ARCH" == "native" ]]; then
    ARCH="$(dpkg --print-architecture)"
fi

case "$ARCH" in
    all|amd64|arm64|armhf|riscv64|i386|ppc64el|s390x) ;;
    *) echo "unsupported architecture: $ARCH" >&2; exit 2 ;;
esac

[[ -n "$OUT_DIR" ]] || OUT_DIR="$ROOT_DIR/dist"
mkdir -p "$OUT_DIR"

# Building an arm64 package on amd64 is fine for a pure-Python payload, but it is
# still worth warning that the result was not executed here.
HOST_ARCH="$(dpkg --print-architecture)"
if [[ "$ARCH" != "all" && "$ARCH" != "$HOST_ARCH" && -z "$CHROOT" ]]; then
    echo "note: building a $ARCH package on $HOST_ARCH." >&2
    echo "      The payload is architecture-independent Python, so this is valid," >&2
    echo "      but run 'yolo-pose selftest' after installing on the target." >&2
fi

if [[ -n "$CHROOT" ]]; then
    echo "error: --chroot requires a prepared qemu/binfmt $ARCH chroot." >&2
    echo "       Prepare one with:" >&2
    echo "         sudo apt install qemu-user-static debootstrap binfmt-support" >&2
    echo "         sudo debootstrap --arch=$ARCH --foreign bookworm /srv/chroot-$ARCH" >&2
    echo "         sudo cp /usr/bin/qemu-$ARCH-static /srv/chroot-$ARCH/usr/bin/" >&2
    echo "         sudo chroot /srv/chroot-$ARCH /debootstrap/debootstrap --second-stage" >&2
    echo "       then re-run this script with --chroot." >&2
    exit 3
fi

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/yolo-pose-build.XXXXXX")"
PKG_ROOT="$BUILD_DIR/${PACKAGE}_${VERSION}_${ARCH}"
cleanup() {
    if [[ "$KEEP_BUILD" -eq 1 ]]; then
        echo "staging kept at: $BUILD_DIR"
    else
        rm -rf "$BUILD_DIR"
    fi
}
trap cleanup EXIT

echo "==> staging $PACKAGE $VERSION ($ARCH)"
install -d "$PKG_ROOT/DEBIAN" \
           "$PKG_ROOT/usr/bin" \
           "$PKG_ROOT/usr/lib/python3/dist-packages" \
           "$PKG_ROOT/usr/share/yolo-pose/models" \
           "$PKG_ROOT/usr/share/doc/$PACKAGE" \
           "$PKG_ROOT/usr/share/man/man1" \
           "$PKG_ROOT/usr/share/applications"

# ---- payload ---------------------------------------------------------------- #
cp -r "$ROOT_DIR/src/yolopose" "$PKG_ROOT/usr/lib/python3/dist-packages/"
find "$PKG_ROOT/usr/lib/python3/dist-packages" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$PKG_ROOT/usr/lib/python3/dist-packages" -name '*.pyc' -delete 2>/dev/null || true

cat > "$PKG_ROOT/usr/bin/yolo-pose" <<'PY'
#!/usr/bin/python3
"""Entry point for the yolo-pose command line interface."""
import sys

from yolopose.cli import main

if __name__ == "__main__":
    sys.exit(main())
PY
chmod 0755 "$PKG_ROOT/usr/bin/yolo-pose"

if [[ "$INCLUDE_MODELS" -eq 1 && -f "$ROOT_DIR/models/yolov8n-pose.onnx" ]]; then
    install -m 0644 "$ROOT_DIR/models/yolov8n-pose.onnx" \
        "$PKG_ROOT/usr/share/yolo-pose/models/"
    echo "    bundled models/yolov8n-pose.onnx"
else
    echo "    no default model bundled (run 'yolo-pose models pull' after install)"
fi

# Face models are large; ship a manifest so the CLI can report what is missing.
cat > "$PKG_ROOT/usr/share/yolo-pose/models/README" <<'TXT'
Default pose model: yolov8n-pose.onnx
Face models (optional): run `yolo-pose models pull --face`
TXT

install -m 0644 "$ROOT_DIR/README.md" "$PKG_ROOT/usr/share/doc/$PACKAGE/README.md"
install -m 0644 "$ROOT_DIR/NOTICE" "$PKG_ROOT/usr/share/doc/$PACKAGE/NOTICE"
install -m 0644 "$ROOT_DIR/CHANGELOG.md" "$PKG_ROOT/usr/share/doc/$PACKAGE/CHANGELOG.md"
install -m 0644 "$ROOT_DIR/LICENSE" "$PKG_ROOT/usr/share/doc/$PACKAGE/LICENSE"
install -m 0644 "$ROOT_DIR/debian/copyright" "$PKG_ROOT/usr/share/doc/$PACKAGE/copyright"
gzip -9n -c "$ROOT_DIR/debian/changelog" > "$PKG_ROOT/usr/share/doc/$PACKAGE/changelog.Debian.gz"

# ---- metadata --------------------------------------------------------------- #
INSTALLED_SIZE="$(du -sk "$PKG_ROOT" | cut -f1)"
cat > "$PKG_ROOT/DEBIAN/control" <<EOF
Package: $PACKAGE
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Depends: python3 (>= 3.8), python3-numpy (>= 1.20)
Recommends: python3-pil, python3-onnxruntime, ffmpeg
Suggests: v4l-utils
Installed-Size: $INSTALLED_SIZE
Maintainer: geekmd <geekermd@users.noreply.github.com>
Homepage: https://github.com/geekermd/yolo-pose
Description: human joint (pose) and face recognition with a pure-NumPy ONNX runtime
 yolo-pose detects human body poses and faces using YOLO-family ONNX models
 executed by a dependency-free NumPy inference engine, so the same code runs on
 x86-64, aarch64 and other architectures without compiled wheels.
 .
 Features:
  * ONNX graph parser and 60+ operator kernels implemented in NumPy
  * YOLOv8/YOLO11 pose: 17 COCO keypoints, skeleton rendering, joint angles
  * Face pipeline: SCRFD detection, ArcFace alignment, MobileFaceNet embeddings
    and gallery-based identity matching
  * Live V4L2 camera preview over X11, plus headless frame output
  * CLI (image, camera, joints, face, models, inspect, selftest) and Python API
 .
 The pretrained model weights are downloaded on demand and are covered by their
 upstream licences (AGPL-3.0 for Ultralytics models, non-commercial research for
 InsightFace models); see /usr/share/doc/yolo-pose/NOTICE.
EOF

cat > "$PKG_ROOT/DEBIAN/postinst" <<'SH'
#!/bin/sh
set -e

if [ "$1" = "configure" ]; then
    # Byte-compile so the first run does not pay the cost.
    if command -v python3 >/dev/null 2>&1; then
        python3 -m compileall -q /usr/lib/python3/dist-packages/yolopose \
            >/dev/null 2>&1 || true
    fi

    if [ -d /dev ] && [ -r /proc/devices ]; then
        cat <<'MSG'

yolo-pose installed. Quick start:
  yolo-pose selftest                    verify the installation
  yolo-pose image photo.jpg -o out.png  pose detection on an image
  yolo-pose camera --device 0           live camera preview
  yolo-pose models pull --face          download the optional face models

MSG
    fi
fi

exit 0
SH
chmod 0755 "$PKG_ROOT/DEBIAN/postinst"

cat > "$PKG_ROOT/DEBIAN/prerm" <<'SH'
#!/bin/sh
set -e
# Remove byte-compiled files created by postinst.
rm -rf /usr/lib/python3/dist-packages/yolopose/__pycache__ 2>/dev/null || true
exit 0
SH
chmod 0755 "$PKG_ROOT/DEBIAN/prerm"

cat > "$PKG_ROOT/usr/share/man/man1/yolo-pose.1" <<'MAN'
.TH YOLO-POSE 1 "2026-10-01" "yolo-pose 1.0.0" "User Commands"
.SH NAME
yolo-pose \- human joint (pose) and face recognition with a pure-NumPy ONNX runtime
.SH SYNOPSIS
.B yolo-pose
.RI [ options ] " command " [ command\-options ]
.SH DESCRIPTION
.B yolo-pose
detects human body poses and faces in images and camera streams.  ONNX graphs are
executed by a built-in NumPy inference engine, so no compiled inference library
is required; ONNX Runtime is used automatically when it is installed.
.SH COMMANDS
.TP
.B image
Annotate one or more images with skeletons, boxes and joint angles.
.TP
.B camera
Capture from a V4L2 device and show a live annotated preview.
.TP
.B joints
Print measured joint angles for every detected person.
.TP
.B face
Detect and recognise faces, and enrol new identities into a gallery.
.TP
.B models
List or download pose and face models.
.TP
.B inspect
Show an ONNX graph, its opset and operator support coverage.
.TP
.B selftest
Verify the installation end to end.
.SH EXAMPLES
.nf
yolo-pose models pull
yolo-pose image photo.jpg -o out.png
yolo-pose joints photo.jpg --json -
yolo-pose camera --device 0
yolo-pose face group.jpg --gallery people.npz
.fi
.SH FILES
.TP
.I /usr/share/yolo-pose/models
Bundled or downloaded model weights.
.TP
.I ~/.local/share/yolo-pose/models
Models fetched with
.BR "yolo-pose models pull" .
.SH SEE ALSO
Full documentation: https://github.com/geekermd/yolo-pose
.SH AUTHOR
geekmd.  Licensed MIT; model weights carry their own licences (see NOTICE).
MAN
gzip -9n "$PKG_ROOT/usr/share/man/man1/yolo-pose.1"

cat > "$PKG_ROOT/usr/share/applications/yolo-pose.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=yolo-pose
Comment=Human pose and face recognition from a camera
Exec=yolo-pose camera --device 0
Terminal=true
Categories=Science;Graphics;
Keywords=pose;keypoints;face;yolo;onnx;
DESK

# ---- build ------------------------------------------------------------------ #
DEB_PATH="$OUT_DIR/${PACKAGE}_${VERSION}_${ARCH}.deb"
echo "==> packaging $DEB_PATH"
dpkg-deb --root-owner-group --build "$PKG_ROOT" "$DEB_PATH" >/dev/null

echo
echo "==> done"
dpkg-deb --info "$DEB_PATH" | sed -n '1,12p'
echo
ls -lh "$DEB_PATH"
echo
echo "install with:  sudo apt install $DEB_PATH"
