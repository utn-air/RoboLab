# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class ReachAppleTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=reach_object,
        params={"object": "apple_01", "tolerance": 0.05},
    )


@dataclass
class RotatedReachAppleTask(Task):
    contact_object_list = ["apple_01", "table"]
    scene = import_scene("breakfast_table.usda", contact_object_list)
    terminations = ReachAppleTerminations
    instruction = {
        "default": "ReachApple",
        "vague": "Reach the apple",
        "specific": "Move the robot gripper to a position just above the apple without grasping it",
    }
    episode_steps: int = 50
    attributes = ["reach", "goal"]
    goal = {
        "mode": "reach",
        "object": "apple_01",
        "tolerance": 0.025,
        "drive_steps": 80,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="reach_above_apple",
            conditions={
                "apple_01": [
                    (partial(reach_object, object="apple_01", tolerance=0.04), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
