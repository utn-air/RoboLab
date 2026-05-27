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
from robolab.core.task.predicate_logic import _and
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledPickupKetchupTask" / "status.json"


def angled_pickup_object(
    env,
    object: str,
    surface: str,
    pos_tolerance: float = 0.04,
    angle_tolerance: float = 0.09,
    lift_distance: float = 0.10,
    ee_pose_key: str = "last_ee_pose_3",
    status_path: str | Path | None = None,
    env_id: int | None = None,
):
    return _and(
        angled_reach_object(
            env,
            pos_tolerance=pos_tolerance,
            angle_tolerance=angle_tolerance,
            ee_pose_key=ee_pose_key,
            status_path=status_path,
            env_id=env_id,
        ),
        object_picked_up(
            env,
            object=object,
            surface=surface,
            distance=lift_distance,
            env_id=env_id,
        ),
    )


@configclass
class AngledPickupKetchupTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_pickup_object,
        params={
            "object": "ketchup_bottle",
            "surface": "table",
            "pos_tolerance": 0.04,
            "angle_tolerance": 0.09,
            "lift_distance": 0.10,
            "status_path": STATUS_PATH,
            "ee_pose_key": "last_ee_pose_3",
        },

    )


@dataclass
class AngledPickupKetchupTask(Task):
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
    terminations = AngledPickupKetchupTerminations
    instruction = {
        "default": "AngledPickupKetchup",
        "vague": "Reach the ketchup bottle with a yawed wrist, grasp it, and lift it up",
        "specific": "Move the robot gripper to the ketchup bottle with the wrist yawed to face the bottle from the side, grasp the bottle, and lift it off the table",
    }
    episode_steps: int = 30
    angledreach_steps: int = 10
    grasp_steps: int = 10
    pickup_steps: int = 10

    attributes = ["angled_reach", "pickup", "grasp", "lift", "dominant_yaw", "+rz", "goal"]
    goal = {
        "mode": "angled_pickup",
        "object": "ketchup_bottle",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_pickup_ketchup",
            conditions={
                "ketchup_bottle": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.04,
                            angle_tolerance=0.09,
                            status_path=STATUS_PATH,
                        ),
                        1.0,
                    ),
                    (partial(object_grabbed, object="ketchup_bottle"), 1.0),
                    (
                        partial(
                            angled_pickup_object,
                            object="ketchup_bottle",
                            surface="table",
                            pos_tolerance=0.04,
                            angle_tolerance=0.09,
                            lift_distance=0.10,
                            status_path=STATUS_PATH,
                            ee_pose_key="last_ee_pose_3",
                        ),
                        1.0,
                    ),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
