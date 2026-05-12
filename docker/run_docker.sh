#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${ROBOLAB_REGISTRY:-robolab}"
IMAGE_TAG="${1:-$(git rev-parse --short HEAD)}"
ROBOLAB_DOCKER_DIR="$(dirname "$(realpath -s "$0")")"
ROBOLAB_DIR="$(realpath "${ROBOLAB_DOCKER_DIR}/../")"

docker run --rm -it \
    --gpus '"device=1"' \
    --network=host \
    -e ACCEPT_EULA=Y \
    --entrypoint /bin/bash \
    -w /workspace/robolab \
    -v "${ROBOLAB_DIR}:/workspace/robolab" \
    "${IMAGE_NAME}:${IMAGE_TAG}"
