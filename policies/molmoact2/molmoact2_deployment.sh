set -euo pipefail

MOLMOACT2_DIR="${MOLMOACT2_DIR:-/workspace/molmoact2}"
ROBOLAB_DIR="${ROBOLAB_DIR:-/workspace/robolab}"

# Replace these with the activation scripts inside your Apptainer image.
MOLMOACT2_ENV="${MOLMOACT2_ENV:-/opt/venvs/molmoact2/bin/activate}"
ROBOLAB_ENV="${ROBOLAB_ENV:-/opt/venvs/robolab/bin/activate}"

SERVER_BIND_HOST="${SERVER_BIND_HOST:-0.0.0.0}"
SERVER_CHECK_HOST="${SERVER_CHECK_HOST:-127.0.0.1}"

HOST_IP="$(
python-rtx-compat - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
finally:
    sock.close()
PY
)"

REMOTE_HOST="${HOST_IP:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-30}"
SERVER_LOG="${SERVER_LOG:-/anvme/workspace/v106be14-vla/${REMOTE_PORT}.log}"

TASKS=(
    BananaOnPlateTask
    BananasInBinThreeTotalTask
    UnstackRubiksCubeTask
    SauceBottlesCrateTask
    RubiksCubeOrBananaTask
    BananaInBowlTask
    BananasInBinOneMoreTask
    RubiksCubeTask
    RubiksCubeThenBananaTask
    BananasInCrateTask
)

NUM_RUNS="${NUM_RUNS:-1}"
NUM_ENVS="${NUM_ENVS:-1}"

SERVER_PID=""

port_open() {
    (echo >"/dev/tcp/${SERVER_CHECK_HOST}/${REMOTE_PORT}") \
        >/dev/null 2>&1
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))

    echo "Waiting for server on ${SERVER_CHECK_HOST}:${REMOTE_PORT}..."

    until port_open; do
        if [[ -n "${SERVER_PID:-}" ]] &&
            ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "MolmoAct2 server exited before becoming ready."
            return 1
        fi

        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for MolmoAct2 server."
            return 1
        fi

        sleep 2
    done

    echo "MolmoAct2 server is ready."
}

cleanup_server() {
    local elapsed=0

    if [[ -z "${SERVER_PID:-}" ]]; then
        return 0
    fi

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping MolmoAct2 server, PID ${SERVER_PID}..."
        kill "$SERVER_PID" 2>/dev/null || true

        while kill -0 "$SERVER_PID" 2>/dev/null &&
            ((elapsed < SERVER_STOP_TIMEOUT)); do
            sleep 1
            ((++elapsed))
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server did not stop gracefully; killing it."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi

        wait "$SERVER_PID" 2>/dev/null || true
    fi

    SERVER_PID=""
}

start_server() {
    if port_open; then
        echo "Using existing server at ${REMOTE_HOST}:${REMOTE_PORT}."
        SERVER_PID=""
        return 0
    fi

    echo "No existing server found. Starting MolmoAct2 server..."

    (
        cd "$MOLMOACT2_DIR"
        source "$MOLMOACT2_ENV"

        exec python examples/droid/host_server_droid.py \
            --host "$SERVER_BIND_HOST" \
            --port "$REMOTE_PORT" \
            --dtype bfloat16
    ) >"$SERVER_LOG" 2>&1 &

    SERVER_PID="$!"

    if ! wait_for_server; then
        echo "MolmoAct2 server failed to become ready."
        tail -n 100 "$SERVER_LOG" 2>/dev/null || true
        return 1
    fi
}

run_simulation() {
    local task

    cd "$ROBOLAB_DIR"

    source "$ROBOLAB_ENV"

    for task in "${TASKS[@]}"; do
        echo
        echo "=== Running RoboLab task: $task ==="

        python-rtx-compat policies/molmoact2/run.py \
            --headless \
            --remote-host "$REMOTE_HOST" \
            --remote-port "$REMOTE_PORT" \
            --task "$task" \
            --num-runs "$NUM_RUNS" \
            --num-envs "$NUM_ENVS" \
            --record-image-data

        echo "=== Completed task: $task ==="
    done
}

trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server
run_simulation

echo "Simulation completed successfully."