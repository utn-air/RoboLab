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
class ReachPitcherTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=reach_object,
        params={"object": "pitcher", "tolerance": 0.05},
    )


@dataclass
class ReachPitcherTask(Task):
    contact_object_list = ["pitcher", "table"]
    scene = import_scene("breakfast_table.usda", contact_object_list)
    terminations = ReachPitcherTerminations
    instruction = {
        "default": "ReachPitcher",
        "vague": "Reach the pitcher",
        "specific": "Move the robot gripper to a position just above the pitcher without grasping it",
    }
    episode_steps: int = 50
    attributes = ["reach", "goal"]
    goal = {
        "mode": "reach",
        "object": "pitcher",
        "tolerance": 0.025,
        "drive_steps": 80,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="reach_above_pitcher",
            conditions={
                "pitcher": [
                    (partial(reach_object, object="pitcher", tolerance=0.04), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
