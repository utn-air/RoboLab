#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ROBOLAB_DIR=$( cd $( dirname ${BASH_SOURCE[0]} )/.. && pwd )

IMAGE_NAME="${ROBOLAB_REGISTRY:-robolab}"
IMAGE_TAG="${1:-$(git rev-parse --short HEAD)}"

XHOST_ENABLED=false
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
  if timeout 5s xhost +local:root >/dev/null 2>&1; then
    XHOST_ENABLED=true
  else
    echo "Warning: X11 display ${DISPLAY} is unavailable; continuing without X11 access." >&2
  fi
fi

cleanup() {
  if [ "$XHOST_ENABLED" = true ]; then
    xhost -local:root >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

docker run \
  -it \
  --entrypoint /bin/bash \
  -e DISPLAY \
  --net host \
  --rm \
  --gpus all \
  -v $ROBOLAB_DIR/.cache/ov:/root/.cache/ov \
  -v $ROBOLAB_DIR/.cache/kit:/isaac-sim/kit/cache \
  -v $ROBOLAB_DIR:/workspace/robolab \
  -w /workspace/robolab \
  "${IMAGE_NAME}:${IMAGE_TAG}"
# xhost -local:root
