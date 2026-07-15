from __future__ import annotations

from typing import Any

import numpy as np
import requests
import json_numpy
from robolab.eval.base_client import InferenceClient

json_numpy.patch()


class MolmoAct2Client(InferenceClient):
    """
    RoboLab client for the MolmoAct2-DROID FastAPI server.

    RoboLab obs:
      raw_obs["image_obs"]["over_shoulder_left_camera"]
      raw_obs["image_obs"]["wrist_cam"]
      raw_obs["proprio_obs"]["arm_joint_pos"]    # 7
      raw_obs["proprio_obs"]["gripper_pos"]      # 1

    MolmoAct2-DROID /act payload:
      external_cam: H x W x 3 uint8
      wrist_cam:    H x W x 3 uint8
      instruction:  str
      state:        shape (8,) float32 = [q1..q7, gripper]
      num_steps:    int
    """

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 10,
        request_timeout: float = 60.0,
        endpoint: str = "/act",
    ) -> None:
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon)
        self.request_timeout = float(request_timeout)

        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.url = f"http://{remote_host}:{remote_port}{endpoint}"
        self.session = requests.Session()

        print(
            f"[MolmoAct2Client] Connecting to MolmoAct2 server: "
            f"{self.url}, open_loop_horizon={self.open_loop_horizon}"
        )

        # Health check. The MolmoAct2 server implements GET /act.
        r = self.session.get(self.url, timeout=self.request_timeout)
        if r.status_code != 200:
            raise RuntimeError(
                f"MolmoAct2 health check failed: {r.status_code}: {r.text[:500]}"
            )
        print(f"[MolmoAct2Client] Server health: {r.text[:300]}")

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        proprio_obs = raw_obs["proprio_obs"]

        external_image = self._to_uint8_hwc(
            image_obs["over_shoulder_left_camera"][env_id]
        )
        wrist_image = self._to_uint8_hwc(image_obs["wrist_cam"][env_id])

        joint_position = self._to_numpy(
            proprio_obs["arm_joint_pos"][env_id]
        ).astype(np.float32)

        gripper_position = self._to_numpy(
            proprio_obs["gripper_pos"][env_id]
        ).astype(np.float32).reshape(-1)

        state = np.concatenate(
            [joint_position.reshape(-1), gripper_position[:1]],
            axis=0,
        ).astype(np.float32)

        if state.shape != (8,):
            raise ValueError(f"Expected MolmoAct2-DROID state shape (8,), got {state.shape}")

        return {
            "external_cam": external_image,
            "wrist_cam": wrist_image,
            "state": state,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "external_cam": extracted_obs["external_cam"],
            "wrist_cam": extracted_obs["wrist_cam"],
            "instruction": instruction,
            "state": extracted_obs["state"],
            "num_steps": self.open_loop_horizon,
        }

    def _query_server(self, request: dict) -> Any:
        r = self.session.post(
            self.url,
            headers={"Content-Type": "application/json"},
            data=json_numpy.dumps(request),
            timeout=self.request_timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"MolmoAct2 server error {r.status_code}: {r.text[:1000]}"
            )

        data = json_numpy.loads(r.text)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"MolmoAct2 inference error: {data['error']}")
        return data

    def _unpack_response(self, response: Any) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)

        if actions.ndim == 1:
            actions = actions[None, :]

        if actions.shape[-1] != 8:
            raise ValueError(f"Expected action dim 8, got {actions.shape}")

        return actions

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        return np.concatenate(
            [extracted_obs["external_cam"], extracted_obs["wrist_cam"]],
            axis=1,
        )

    def close(self) -> None:
        self.session.close()

    @staticmethod
    def _to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    @classmethod
    def _to_uint8_hwc(cls, x) -> np.ndarray:
        arr = cls._to_numpy(x)

        # Common RoboLab image format is HWC. Keep CHW fallback just in case.
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"Expected image HxWx3, got {arr.shape}")

        arr = arr[..., :3]

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr