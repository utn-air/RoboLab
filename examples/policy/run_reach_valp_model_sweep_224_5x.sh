#!/usr/bin/env bash
set -euo pipefail

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/robolab/output}"
HEADLESS="${HEADLESS:-1}"
VIDEO_MODE="${VIDEO_MODE:-sensor}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$OUTPUT_ROOT/valp_model_sweep_logs_${REMOTE_PORT}}"
ARCHIVE_AFTER_MODEL="${ARCHIVE_AFTER_MODEL:-1}"
DELETE_UNZIPPED_AFTER_ARCHIVE="${DELETE_UNZIPPED_AFTER_ARCHIVE:-1}"

MODEL_CONFIGS=(
    droid-224px-8f-dual.yaml
    droid-224px-8f-ind.yaml
    droid-224px-8f-right.yaml
    droid-224px-8f-roboarena.yaml
    droid-224px-8f-wrist.yaml
)

TASKS=(
    ReachCoffeePotTask
    ReachAppleTask
    ReachBagelTask
    ReachCeramicMugTask
    ReachCoffeeCanTask
    ReachForkBigTask
    ReachOrangeTask
    ReachMilkCartonTask
    ReachSpoonBigTask
    ReachYogurtCupTask
)

SERVER_PID=""
MODEL_NAMES=()

port_open() {
    (echo >"/dev/tcp/$REMOTE_HOST/$REMOTE_PORT") >/dev/null 2>&1
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))

    until port_open; do
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return 1
        fi
        if (( SECONDS >= deadline )); then
            return 1
        fi
        sleep 5
    done
}

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

hosted_model_name() {
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    REMOTE_HOST="$REMOTE_HOST" \
    REMOTE_PORT="$REMOTE_PORT" \
    "$ISAAC_PYTHON" -c '
import os
import pickle
import socket
import struct

header = struct.Struct("!Q")
host = os.environ["REMOTE_HOST"]
port = int(os.environ["REMOTE_PORT"])
payload = pickle.dumps({"method": "metadata"}, protocol=pickle.HIGHEST_PROTOCOL)

with socket.create_connection((host, port), timeout=30) as sock:
    sock.sendall(header.pack(len(payload)) + payload)
    size_data = b""
    while len(size_data) < header.size:
        chunk = sock.recv(header.size - len(size_data))
        if not chunk:
            raise ConnectionError("VALP policy socket closed while reading metadata header")
        size_data += chunk
    size = header.unpack(size_data)[0]
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("VALP policy socket closed while reading metadata payload")
        data += chunk

response = pickle.loads(data)
if error := response.get("error"):
    raise RuntimeError(error)
print(response["modelname"])
'
}

archive_model_output() {
    local model_name="$1"
    local model_dir="$OUTPUT_ROOT/$model_name"
    local archive_file="$OUTPUT_ROOT/${model_name}.zip"

    if [[ "$ARCHIVE_AFTER_MODEL" != "1" ]]; then
        return 0
    fi

    if [[ ! -d "$model_dir" ]]; then
        echo "Expected output folder not found, skipping archive: $model_dir"
        return 1
    fi


    echo "=== Zipping $model_dir -> $archive_file ==="
    rm -f "$archive_file"
    (
        cd "$OUTPUT_ROOT"
        zip -qr "$(basename "$archive_file")" "$model_name"
    )

    if [[ "$DELETE_UNZIPPED_AFTER_ARCHIVE" == "1" ]]; then
        echo "=== Removing expanded output folder after successful zip: $model_dir ==="
        rm -rf "$model_dir"
    fi
}

trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT
trap 'cleanup_server; exit 143' TERM

if port_open; then
    echo "Port $REMOTE_HOST:$REMOTE_PORT is already open."
    echo "Stop the existing VALP server before running the model sweep, so each cfg is evaluated against the intended hosted model."
    exit 1
fi

mkdir -p "$SERVER_LOG_DIR"

for cfg_file in "${MODEL_CONFIGS[@]}"; do
    cfg_name="${cfg_file%.yaml}"
    server_log="$SERVER_LOG_DIR/${cfg_name}_serve_policy.log"

    echo
    echo "=== Starting VALP server: $cfg_file ==="
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "$ISAAC_PYTHON" valp/inference/serve_policy.py \
        --cfg-file "$cfg_file" \
        --host "$SERVER_HOST" \
        --port "$REMOTE_PORT" \
        >"$server_log" 2>&1 &
    SERVER_PID="$!"

    if ! wait_for_server; then
        echo "VALP server failed to become ready for $cfg_file."
        echo "Last server log lines from $server_log:"
        tail -n 80 "$server_log" || true
        exit 1
    fi

    model_name="$(hosted_model_name)"
    MODEL_NAMES+=("$model_name")

    echo "=== Running 50 eval episodes for hosted model $model_name ==="
    REMOTE_HOST="$REMOTE_HOST" \
    REMOTE_PORT="$REMOTE_PORT" \
    HEADLESS="$HEADLESS" \
    VIDEO_MODE="$VIDEO_MODE" \
    OUTPUT_FOLDER_NAME="$OUTPUT_FOLDER_NAME" \
        bash examples/policy/run_reach_valp_eval_5x.sh

    archive_model_output "$model_name"
    cleanup_server
done
