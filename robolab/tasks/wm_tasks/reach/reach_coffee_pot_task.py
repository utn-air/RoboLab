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
class ReachCoffeePotTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=reach_object,
        params={"object": "coffee_pot", "tolerance": 0.08},
    )


@dataclass
class ReachCoffeePotTask(Task):
    contact_object_list = ["table, bowl, banana, bagel_07, coffee_can, banana_01, yogurt_cup, coffee_pot, ceramic_mug, pitcher, fork_big, spoon_big, apple_01, orange2, milk_carton, orange_juice_carton, bagel_01, bagel_02, plate_small, plate_large"]
    scene = import_scene("breakfast_table.usda", contact_object_list)
    terminations = ReachCoffeePotTerminations
    instruction = {
        "default": "ReachCoffeePot",
        "vague": "Reach the coffee pot",
        "specific": "Move the robot gripper to a position just above the coffee pot without grasping it",
    }
    episode_steps: int = 50
    attributes = ["reach", "goal"]
    goal = {
        "mode": "reach",
        "object": "coffee_pot",
        "tolerance": 0.025,
        "drive_steps": 80,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="reach_above_coffee_pot",
            conditions={
                "coffee_pot": [
                    (partial(reach_object, object="coffee_pot", tolerance=0.08), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
