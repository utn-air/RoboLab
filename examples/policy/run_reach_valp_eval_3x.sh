#!/usr/bin/env bash
set -euo pipefail

ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

TASKS=(
    ReachAppleTask
    ReachBagelTask
    ReachCeramicMugTask
    ReachCoffeeCanTask
    ReachCoffeePotTask
    ReachForkBigTask
    ReachOrangeTask
    ReachPitcherTask
    ReachSpoonBigTask
    ReachYogurtCupTask
)

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

MODEL_NAME="$(hosted_model_name)"

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$ISAAC_PYTHON" examples/policy/run_eval.py \
    --policy valp \
    --num-runs 3 \
    --num-envs 1 \
    --task "${TASKS[@]}" \
    --remote-host "$REMOTE_HOST" \
    --remote-port "$REMOTE_PORT"

"$ISAAC_PYTHON" analysis/summarize_eval_success.py \
    "robolab/output/$MODEL_NAME" \
    --expected-runs 3 \
    --task "${TASKS[@]}"
