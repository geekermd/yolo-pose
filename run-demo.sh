#!/usr/bin/env bash
#
# Start the local browser demo (no camera needed).
#
#   ./run-demo.sh                 # http://127.0.0.1:8765
#   ./run-demo.sh --open          # open the browser automatically
#   ./run-demo.sh --port 9000

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ -d "$HERE/src/yolopose" && -z "${FORCE_INSTALLED:-}" ]]; then
    export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
    RUNNER=(python3 -m yolopose)
else
    RUNNER=(yolo-pose)
fi

exec "${RUNNER[@]}" demo "${@}"
