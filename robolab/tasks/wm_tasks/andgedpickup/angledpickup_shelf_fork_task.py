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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupShelfForkTask" / "status.json"


@configclass
class AngledPickupShelfForkTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": "fork_big", "surface": "sm_rack_m01", "distance": 0.05},
    )


@dataclass
class AngledPickupShelfForkTask(Task):
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
    terminations = AngledPickupShelfForkTerminations
    instruction = {
        "default": "AngledPickupShelfFork",
        "vague": "Reach the fork on the upper shelf with a pitched wrist, grasp it, and lift it up",
        "specific": "Move the robot gripper toward the fork on the upper shelf with the wrist pitched down into the shelf opening, grasp the fork, and lift it away from the shelf",
    }
    episode_steps: int = 80
    attributes = ["angled_reach", "pickup", "grasp", "lift", "dominant_yaw_pitch", "+rz-ry", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "fork_big",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_fork_big",
            conditions={
                "fork_big": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.10,
                            angle_tolerance=0.20,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    ),
                    (partial(object_grabbed, object="fork_big"), 1.0),
                    (
                        partial(
                            object_picked_up,
                            object="fork_big",
                            surface="sm_rack_m01",
                            distance=0.05,
                        ),
                        1.0,
                    ),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
