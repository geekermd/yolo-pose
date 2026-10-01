#!/usr/bin/env bash
#
# Ask which live mode to start, then launch it.
# Used by the desktop/taskbar entries so both features stay available.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
    if command -v zenity >/dev/null 2>&1; then
        MODE=$(zenity --list --title="yolo-pose 实时识别" \
            --text="请选择要启动的实时识别功能：\n（识别窗口启动后可自由拖拽缩放）" \
            --column="模式" --column="说明" \
            pose      "人体姿态识别（17 关节 + 骨架 + 关节角度）" \
            face      "人脸识别（检测 + 身份匹配）" \
            both      "姿态 + 人脸（两者同时）" \
            both-big  "姿态 + 人脸 —— 大窗口（2 倍，1280x960）" \
            both-huge "姿态 + 人脸 —— 超大窗口（3 倍，1920x1440）" \
            demo      "浏览器演示（无需摄像头）" \
            --width=560 --height=340 2>/dev/null) || exit 0
    else
        echo "yolo-pose 实时识别 — 请选择模式："
        echo "  1) pose      人体姿态识别（17 关节 + 骨架 + 关节角度）"
        echo "  2) face      人脸识别（检测 + 身份匹配）"
        echo "  3) both      姿态 + 人脸（两者同时）"
        echo "  4) both-big  姿态 + 人脸 —— 大窗口 2 倍"
        echo "  5) both-huge 姿态 + 人脸 —— 超大窗口 3 倍"
        echo "  6) demo      浏览器演示（无需摄像头）"
        echo "  （窗口启动后可自由拖拽缩放）"
        read -r -p "输入 1-6 [默认 3]: " choice || exit 0
        case "${choice:-3}" in
            1) MODE=pose ;; 2) MODE=face ;; 4) MODE=both-big ;;
            5) MODE=both-huge ;; 6) MODE=demo ;; *) MODE=both ;;
        esac
    fi
fi

case "$MODE" in
    pose)      exec "$HERE/run-camera.sh" --pose-only ;;
    face)      exec "$HERE/run-camera.sh" --face-only ;;
    both)      exec "$HERE/run-camera.sh" ;;
    both-big)  exec "$HERE/run-camera.sh" --big ;;
    both-huge) exec "$HERE/run-camera.sh" --huge ;;
    demo)      exec "$HERE/run-demo.sh" ;;
    *)         echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
