# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world


def gripper_above_object(
    env,
    object: str,
    z_offset: float = 0.12,
    tolerance: float = 0.04,
    link_name: str = "base_link",
    env_id: int | None = None,
):
    world = get_world(env)

    if env_id is not None:
        corners, centroid = world.get_bbox(object, env_id=env_id)
        target = torch.tensor(centroid, dtype=torch.float32, device=env.device)
        target[2] = max(corner[2] for corner in corners) + z_offset
        target = target + env.scene.env_origins[env_id]
        gripper_pos = world.get_articulation_link_pose("robot", link_name, env_id=env_id)[:3]
        return torch.linalg.norm(gripper_pos - target).item() <= tolerance

    corners, centroid = world.get_bbox(object, env_id=None)
    target = centroid.clone()
    target[:, 2] = corners[:, :, 2].max(dim=1).values + z_offset
    target = target + env.scene.env_origins
    gripper_pos = world.get_articulation_link_pose("robot", link_name, env_id=None)[:, :3]
    return torch.linalg.norm(gripper_pos - target, dim=1) <= tolerance


@configclass
class ReachBananaTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=gripper_above_object,
        params={"object": "banana", 
                "z_offset": 0.12, 
                "tolerance": 0.04
                },
    )


@dataclass
class ReachBananaTask(Task):
    contact_object_list = ["banana", "bowl", "table"]
    scene = import_scene("banana_bowl.usda", contact_object_list) #rubiks_cube_banana_bowl.usda
    terminations = ReachBananaTerminations
    instruction = {
        "default": "ReachBanana",
        "vague": "Reach the fruit",
        "specific": "Move the robot gripper to a position just above the yellow banana without grasping it",
    }
    # episode_length_s: int = 20
    episode_steps: int = 25
    attributes = ["reach", "goal"]
    goal = {
        "mode": "reach_above_object",
        "object": "banana",
        "z_offset": 0.12,
        "tolerance": 0.025,
        "drive_steps": 80,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="reach_above_banana",
            conditions={
                "banana": [
                    (partial(gripper_above_object, object="banana", z_offset=0.12, tolerance=0.04), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
