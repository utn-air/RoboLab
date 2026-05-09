# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import angled_reach_object
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachKetchupTask" / "status.json"

@configclass
class AngledReachKetchupTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"object": "ketchup_bottle", "tolerance": 0.05, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachKetchupTask(Task):
    contact_object_list = [
        "table",
        "sm_rack_m01",
        "rack_l04",
        "ketchup_bottle",
        "mustard",
    ]
    scene = import_scene("shelf2_with_condiments.usda", contact_object_list)
    terminations = AngledReachKetchupTerminations
    instruction = {
        "default": "AngledReachKetchup",
        "vague": "Reach the ketchup bottle from the side",
        "specific": "Move the robot gripper to a position next to the ketchup facing the bottle without grasping it",
    }
    episode_steps: int = 50
    attributes = ["angled_reach", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "ketchup_bottle",
        "drive_steps": 30,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_ketchup",
            conditions={
                "ketchup_bottle": [
                    (partial(angled_reach_object, object="ketchup_bottle", status_path=STATUS_PATH), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
