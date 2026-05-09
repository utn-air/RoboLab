# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class HoldRedHammerTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_grabbed,
        params={"object": "red_hammer"},
    )


@dataclass
class HoldRedHammerTask(Task):
    contact_object_list = [
        "table",
        "clamp",
        "cordless_drill",
        "spring_clamp",
        "husky_hammer",
        "right_bin",
        "center_bin",
        "left_bin",
        "blue_hammer",
        "red_hammer",
        "wood_hammer",
    ]
    scene = import_scene("tools_sorting.usda", contact_object_list)
    terminations = HoldRedHammerTerminations
    instruction = {
        "default": "HoldRedHammer",
        "vague": "Hold the red hammer",
        "specific": "Grasp the red hammer and keep it in the gripper",
    }
    episode_steps: int = 60
    attributes = ["grasp", "goal", "tools"]
    goal = {
        "mode": "reach",
        "object": "red_hammer",
        "tolerance": 0.07,
        "drive_steps": 80,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="hold_red_hammer",
            conditions={
                "red_hammer": [
                    (partial(reach_object, object="red_hammer", tolerance=0.07), 0.5),
                    (partial(object_grabbed, object="red_hammer"), 1.0),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
