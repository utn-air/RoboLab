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
from robolab.core.task.conditionals import angled_reach_object, object_grabbed, object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupMilkCartonTask" / "status.json"


@configclass
class AngledPickupMilkCartonTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": "milk_carton", "surface": "table", "distance": 0.30},
    )


@dataclass
class AngledPickupMilkCartonTask(Task):
    contact_object_list = ["table", "milk_carton"]
    scene = import_scene("milk_carton_center_table.usda", contact_object_list)
    terminations = AngledPickupMilkCartonTerminations
    instruction = {
        "default": "AngledPickupMilkCarton",
        "vague": "Reach the milk carton with a pitched wrist, grasp it, and lift it up",
        "specific": "Move the robot gripper to the front of the milk carton with the wrist pitched, grasp the carton, and lift it off the table",
    }
    episode_steps: int = 80
    attributes = ["angled_reach", "pickup", "grasp", "lift", "dominant_pitch", "-ry", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "milk_carton",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_milk_carton",
            conditions={
                "milk_carton": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.10,
                            angle_tolerance=0.35,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    ),
                    (partial(object_grabbed, object="milk_carton"), 1.0),
                    (
                        partial(
                            object_picked_up,
                            object="milk_carton",
                            surface="table",
                            distance=0.30,
                        ),
                        1.0,
                    ),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
