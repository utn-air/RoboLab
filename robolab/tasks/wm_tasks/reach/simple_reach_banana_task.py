# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import ASSET_DIR
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "SimpleReachBananaTask" / "status.json"


@configclass
class ReachBananaTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=reach_object,
        params={"object": "banana", "tolerance": 0.05, "status_path": STATUS_PATH},
    )


@dataclass
class SimpleReachBananaTask(Task):
    contact_object_list = ["table", "bowl", "banana"]
    scene = import_scene("banana_bowl.usda", contact_object_list)
    terminations = ReachBananaTerminations
    instruction = {
        "default": "ReachBanana",
        "vague": "Reach the fruit",
        "specific": "Move the robot gripper to a position just above the yellow banana without grasping it",
    }
    # episode_length_s: int = 20
    episode_steps: int = 50
    attributes = ["reach", "goal"]
    goal = {
        "mode": "reach",
        "object": "banana",
        "z_offset": 0.15,
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
                    (partial(reach_object, object="banana", tolerance=0.05, status_path=STATUS_PATH), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
