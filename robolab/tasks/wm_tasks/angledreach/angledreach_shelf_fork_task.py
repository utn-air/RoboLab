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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachShelfForkTask" / "status.json"

@configclass
class AngledReachShelfForkTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
                "pos_tolerance": 0.10, 
                "angle_tolerance": 0.35, 
                "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachShelfForkTask(Task):
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
    terminations = AngledReachShelfForkTerminations
    instruction = {
        "default": "AngledReachShelfFork",
        "vague": "Reach the fork on the upper shelf with a pitched approach",
        "specific": "Move the robot gripper toward the fork on the upper shelf with the wrist pitched down into the shelf opening, facing the handle without grasping it",
    }
    episode_steps: int = 50
    attributes = ["angled_reach", "dominant_yaw_pitch", "+rz-ry", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "fork_big",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_fork_big",
            conditions={
                "fork_big": [
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
