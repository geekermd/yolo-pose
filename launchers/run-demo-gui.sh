#!/usr/bin/env bash
# Generated launcher wrapper — runs the tool and then pauses so a terminal
# window stays open long enough to read any error message.
#
# The project root is baked in as an absolute path: computing it from
# BASH_SOURCE is fragile when the script is started by a desktop environment.

set -u
ROOT="/home/md/桌面/yolo-pose"

if [ ! -d "$ROOT" ]; then
    echo "找不到项目目录：$ROOT" >&2
    read -r -p "按回车键关闭…" _
    exit 1
fi

cd "$ROOT" || exit 1
./run-demo.sh --open

status=$?
echo
if [ $status -ne 0 ]; then
    echo "退出码 $status —— 请把上面的信息截图反馈。"
fi
read -r -p "按回车键关闭此窗口…" _
exit $status
