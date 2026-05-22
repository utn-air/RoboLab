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
from robolab.core.task.conditionals import angled_reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachShelfTask" / "status.json"

@configclass
class AngledReachShelfTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"pos_tolerance": 0.10, 
                "angle_tolerance": 0.35, 
                "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachShelfTask(Task):
    contact_object_list = [
        "table",
        "sm_rack_m01",
        "spatula_05",
        "spatula_14",
        "spoon_big",
        "spoon_small",
        "fork_big",
        "fork_small",
    ]
    scene = import_scene("cutlery_shelf.usda", contact_object_list)
    terminations = AngledReachShelfTerminations
    instruction = {
        "default": "AngledReachShelf",
        "vague": "Reach the upper shelf with a rolled approach from right side",
        "specific": "Move the robot gripper toward the right side of the upper shelf with the wrist rolled so the fingers align vertically with the shelf bar, without grasping it",
    }
    episode_steps: int = 50
    attributes = ["angled_reach", "dominant_roll", "+rx", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "sm_rack_m01",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_shelf",
            conditions={
                "sm_rack_m01": [
                    (partial(angled_reach_object, 
                            pos_tolerance=0.10, 
                            angle_tolerance=0.35, 
                            status_path=STATUS_PATH),
                    1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
