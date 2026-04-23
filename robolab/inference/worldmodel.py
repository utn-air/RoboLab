# my_policy/inference_client.py

import numpy as np
from robolab.inference.base_client import InferenceClient


class MyPolicyClient(InferenceClient):
    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000) -> None:
        # Connect to your model server
        ...

    def infer(self, obs: dict, instruction: str) -> dict:
        # For the default DROID registration, obs contains:
        #   obs["image_obs"]["external_cam"]    - (N, H, W, 3) torch tensor, uint8
        #   obs["image_obs"]["wrist_cam"]       - (N, H, W, 3) torch tensor, uint8
        #   obs["proprio_obs"]["arm_joint_pos"] - (N, 7) torch tensor, float32
        #   obs["proprio_obs"]["gripper_pos"]   - (N, 1) torch tensor, float32

        # Extract observations for this env (N = num_envs; index by env_id)
        image = obs["image_obs"]["external_cam"][0].cpu().numpy()
        joint_pos = obs["proprio_obs"]["arm_joint_pos"][0].cpu().numpy()

        # Call your model server and get back an action
        action = self._query_server(image, joint_pos, instruction)

        # Return dict with "action" (np.ndarray) and "viz" (np.ndarray for display)
        return {
            "action": action,  # shape (8,): 7 joint positions + 1 gripper {0, 1}
            "viz": image,      # any RGB image for the live visualization window
        }

    def reset(self):
        # Called between episodes. Clear any internal state (action buffers, etc.)
        ...