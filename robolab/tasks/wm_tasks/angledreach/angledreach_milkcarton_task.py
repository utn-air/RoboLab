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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachMilkCartonRollTask" / "status.json"


@configclass
class AngledReachMilkCartonTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
            "pos_tolerance": 0.10,
            "angle_tolerance": 0.20,
            "status_path": STATUS_PATH,
        },
    )


@dataclass
class AngledReachMilkCartonTask(Task):
    contact_object_list = ["table", "milk_carton"]
    scene = import_scene("milk_carton_center_table.usda", contact_object_list)
    terminations = AngledReachMilkCartonTerminations
    instruction = {
        "default": "AngledReachMilkCarton",
        "vague": "Reach the tall carton in the middle of the table from the front face with a pitched wrist",
        "specific": "Move the robot gripper to the front of the milk carton in the middle of the table with the wrist pitched, without grasping it",
    }
    episode_steps: int = 60
    attributes = ["angled_reach", "dominant_pitch", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "milk_carton",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_milk_carton",
            conditions={
                "milk_carton": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.10,
                            angle_tolerance=0.20,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    )
                ]
            },
            logical="all",
            score=1.0,
        )
    ]