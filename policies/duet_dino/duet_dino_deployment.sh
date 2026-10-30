#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --task TASK [--cfg-file CONFIG]" >&2
}

TASK=""
CFG_FILE="angledpickup/droid-224px-8f-dual.yaml"
REMOTE_PORT="${REMOTE_PORT:-8000}"
DUET_DINO_DIR="${DUET_DINO_DIR:-/workspace/duet-dino}"
ROBOLAB_DIR="${ROBOLAB_DIR:-/workspace/robolab}"
DUET_DINO_PYTHON="${DUET_DINO_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
ROBOLAB_PYTHON_WRAPPER="${ROBOLAB_PYTHON_WRAPPER:-$ROBOLAB_DIR/docker/vulkan-rtx-compat/python-rtx-compat}"
export ROBOLAB_PYTHON="${ROBOLAB_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"

while (($#)); do
    case "$1" in
        --task) TASK="${2:?Missing task name}"; shift 2 ;;
        --cfg-file) CFG_FILE="${2:?Missing config file}"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ -z "$TASK" ]]; then
    usage
    exit 2
fi

SERVER_LOG="${SERVER_LOG:-/tmp/duet_dino_${REMOTE_PORT}.log}"
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$DUET_DINO_DIR"
"$DUET_DINO_PYTHON" inference/serve_policy_pickup.py \
    --cfg-file "$CFG_FILE" --host 127.0.0.1 --port "$REMOTE_PORT" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for ((waited=0; waited<600; waited+=2)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "DUET-DINO server exited; see $SERVER_LOG" >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if (echo >/dev/tcp/127.0.0.1/"$REMOTE_PORT") 2>/dev/null; then
        break
    fi
    sleep 2
done
if ! (echo >/dev/tcp/127.0.0.1/"$REMOTE_PORT") 2>/dev/null; then
    echo "Timed out waiting for DUET-DINO server; see $SERVER_LOG" >&2
    exit 1
fi

cd "$ROBOLAB_DIR"
"$ROBOLAB_PYTHON_WRAPPER" policies/duet_dino/run.py \
    --headless --task "$TASK" \
    --remote-host 127.0.0.1 --remote-port "$REMOTE_PORT" \
    --video-mode sensor --instruction-type vague
