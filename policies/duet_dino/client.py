from __future__ import annotations

import time
import pickle
import socket
import struct
import traceback
from pathlib import Path

import numpy as np
import cv2
import torch

from robolab.eval.base_client import InferenceClient

################################ CLIENT ############################################

_HEADER = struct.Struct("!Q")

def _send_message(sock: socket.socket, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)

def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("DUET-DINO policy socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def _recv_message(sock: socket.socket) -> dict:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))

class DUETDINOClient(InferenceClient):
    """Local RoboLab inference client for the DUET-DINO world model on DroidIK envs."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        task_dir: str | Path | None = None,
        run: int = 0,
        key_frames: int = 10,
        planning_steps: int = 20,
    ) -> None:
        super().__init__()
        if key_frames < 1 or planning_steps < 1 or run < 0:
            raise ValueError("run must be nonnegative; key_frames and planning_steps must be positive")
        self.run = run
        self.planning_steps = planning_steps
        self.goals = self._load_goals(Path(task_dir), run, key_frames) if task_dir else None
        self._step_by_env: dict[int, int] = {}
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.sock = self._connect()

    @staticmethod
    def _load_goals(task_dir: Path, run: int, key_frames: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        videos = sorted(
            path for path in task_dir.glob(f"*_{run}.mp4")
            if not path.name.endswith("_viewport.mp4")
        )
        if len(videos) != 1:
            raise ValueError(f"Expected one sensor video for run {run} in {task_dir}, found {videos}")
        video = cv2.VideoCapture(str(videos[0]))
        try:
            if not video.isOpened():
                raise ValueError(f"Cannot open video {videos[0]}")
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < key_frames:
                raise ValueError(f"Video has {frame_count} frames; requested {key_frames} distinct goals")
            indices = np.linspace(0, frame_count - 1, key_frames, dtype=int)
            goals = []
            for index in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, bgr = video.read()
                if not ok:
                    raise ValueError(f"Could not decode frame {index} of {videos[0]}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[1] % 2:
                    raise ValueError(f"Expected two equally sized camera views in {videos[0]}")
                external, wrist = np.split(rgb, 2, axis=1)
                goals.append((torch.from_numpy(external.copy()), torch.from_numpy(wrist.copy())))
            print(f"[DUETDINOClient] Goals from {videos[0]} at frames {indices.tolist()}")
            return goals
        finally:
            video.release()

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self._step_by_env.clear()

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        if self.goals is not None:
            step = self._step_by_env.get(env_id, 0)
            goal_index = min(step // self.planning_steps, len(self.goals) - 1)
            if step % self.planning_steps == 0 and step // self.planning_steps < len(self.goals):
                external, wrist = self.goals[goal_index]
                self.set_goal_images(
                    external, wrist, env_id=env_id,
                    instruction=f"{instruction}_goal{goal_index}", run_idx=self.run,
                )
                print(f"[DUETDINOClient] env {env_id}: goal {goal_index + 1}/{len(self.goals)} at step {step}")
            result = super().infer(obs, instruction, env_id=env_id)
            self._step_by_env[env_id] = step + 1
            return result
        return super().infer(obs, instruction, env_id=env_id)

    def _connect(self) -> socket.socket:
        print(f"[{self.__class__.__name__}] Waiting for DUET-DINO server on {self.remote_host}:{self.remote_port}...")
        while True:
            try:
                sock = socket.create_connection((self.remote_host, self.remote_port), timeout=10)
                sock.settimeout(None)
                print(f"[{self.__class__.__name__}] Connected to DUET-DINO server.")
                return sock
            except OSError:
                time.sleep(2)

    def _request(self, payload: dict) -> dict:
        _send_message(self.sock, payload)
        response = _recv_message(self.sock)
        if error := response.get("error"):
            raise RuntimeError(f"DUET-DINO server error:\n{error}")
        return response

    def reset(self, *, env_id: int | None = None):
        if env_id is None:
            self._request({"method": "reset"})
        super().reset(env_id=env_id)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def set_goal_images(
        self,
        external_image,
        wrist_image,
        *,
        env_id: int = 0,
        instruction: str = "goal",
        run_idx: int | None = None,
    ):
        payload = {
            "method": "set_goal_images",
            "external_image": external_image,
            "wrist_image": wrist_image,
            "env_id": env_id,
            "instruction": instruction,
        }
        if run_idx is not None:
            payload["run_idx"] = int(run_idx)
        self._request(payload)

    def _extract_observation(self, obs_dict: dict, *, env_id: int) -> dict:
        from scipy.spatial.transform import Rotation

        external_image = obs_dict["image_obs"]["over_shoulder_right_camera"][env_id].clone().detach().cpu()
        wrist_image = obs_dict["image_obs"]["wrist_cam"][env_id].clone().detach().cpu()

        robot_state = obs_dict["proprio_obs"]
        # EE position
        ee_pos = robot_state["ee_pos"][env_id].clone().detach().cpu().numpy()
        # EE orientation
        ee_quat = robot_state["ee_quat"][env_id].clone().detach().cpu().numpy()
        ee_rpy = Rotation.from_quat(ee_quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)
        # gripper state
        gripper_pos = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()
        ee_pose = np.concatenate([ee_pos, ee_rpy, gripper_pos], axis=0).astype(np.float32)

        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "ee_pose": ee_pose,
            "env_id": env_id,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "method": "infer",
            "obs": {
                "external_image": extracted_obs["external_image"],
                "wrist_image": extracted_obs["wrist_image"],
                "ee_pose": extracted_obs["ee_pose"],
            },
            "instruction": instruction,
            "env_id": extracted_obs["env_id"],
        }

    def _query_server(self, request: dict) -> dict:
        return self._request(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        action = response.get("action")
        if action is None:
            action = response.get("actions")
        if action is None:
            raise KeyError("DUET-DINO response did not contain 'action' or 'actions'.")

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        return action
    
    def report_ground_truth(self, obs_dict: dict, *, env_id: int = 0) -> dict:
        """Match the post-step simulator observation to the pending imagination."""
        extracted = self._extract_observation(obs_dict, env_id=env_id)
        return self._request({
            "method": "report_ground_truth",
            "obs": {
                "external_image": extracted["external_image"],
                "wrist_image": extracted["wrist_image"],
                "ee_pose": extracted["ee_pose"],
            },
            "env_id": env_id,
        })

    def finish_rollout(self, *, env_id: int = 0) -> dict:
        """Finalize a matched imagination/ground-truth analysis rollout."""
        return self._request({
            "method": "finish_rollout",
            "env_id": env_id,
        })

    def metadata(self):
        return self._request({"method": "metadata"})

MyPolicyClient = DUETDINOClient

