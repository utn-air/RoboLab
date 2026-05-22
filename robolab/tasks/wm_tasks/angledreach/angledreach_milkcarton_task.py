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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachMilkCartonTask" / "status.json"

@configclass
class AngledReachMilkCartonTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
                "pos_tolerance": 0.10, 
                "angle_tolerance": 0.35, 
                "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachMilkCartonTask(Task):
    contact_object_list = [
        "table",
        "alphabet_soup_can",
        "milk_carton",
        "orange_juice_carton",
        "smartphone",
        "mug",
        "mayonnaise_bottle",
        "ketchup_bottle",
        "cubebox_a02",
    ]
    scene = import_scene("cartons_on_box.usda", contact_object_list)
    terminations = AngledReachMilkCartonTerminations
    instruction = {
        "default": "AngledReachMilkCarton",
        "vague": "Reach the milk carton on the packing table with a pitched approach",
        "specific": "Move the robot gripper to the milk carton on the table, pitched to face the carton from the front of the carton, without grasping it",
    }
    episode_steps: int = 50
    attributes = ["angled_reach", "dominant_pitch", "+ry", "goal"]
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
