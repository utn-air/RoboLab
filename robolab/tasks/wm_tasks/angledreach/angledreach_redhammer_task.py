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
from robolab.core.task.conditionals import object_grabbed, reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachRedHammerTask" / "status.json"

@configclass
class AngledReachRedHammerTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"object": "red_hammer", 
                "pos_tolerance": 0.10, 
                "angle_tolerance": 0.20, 
                "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachRedHammerTask(Task):
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
    terminations = AngledReachRedHammerTerminations
    instruction = {
        "default": "AngledReachRedHammer",
        "vague": "Reach the red hammer from the side",
        "specific": "Move the robot gripper to a position next to the red hammer facing it without grasping it",
    }
    episode_steps: int = 60
    attributes = ["angled_reach", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "red_hammer",
        "drive_steps": 30,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_red_hammer",
            conditions={
                "red_hammer": [
                    (partial(angled_reach_object, 
                            object="red_hammer", 
                            pos_tolerance=0.10, 
                            angle_tolerance=0.20, 
                            status_path=STATUS_PATH),
                    1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
