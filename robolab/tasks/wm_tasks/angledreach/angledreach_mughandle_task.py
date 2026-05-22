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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachMugHandleTask" / "status.json"


@configclass
class AngledReachMugHandleTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={
            "pos_tolerance": 0.10,
            "angle_tolerance": 0.35,
            "status_path": STATUS_PATH,
        },
    )


@dataclass
class AngledReachMugHandle(Task):
    contact_object_list = [
        "ceramic_mug",
        "glasses",
        "keyboard",
        "lizard_figurine",
        "marker",
        "remote_control",
        "rubiks_cube",
        "smartphone",
        "wooden_bowl",
        "spoon_big",
        "computer_mouse",
        "yogurt_cup",
        "oatmeal_raisin_cookies",
        "granola_bars",
        "table"
    ]
    scene = import_scene("workdesk.usda", contact_object_list)
    terminations = AngledReachMugHandleTerminations
    instruction = {
        "default": "AngledReachMugHandle",
        "vague": "Reach the Handle of the mug with a yaw-rolled wrist",
        "specific": "Move the robot gripper next to the mug handle with the wrist yaw-rolled so the fingers align vertically with the thin mug handle, without grasping it",
    }
    episode_steps: int = 50
    attributes = ["angled_reach", "yaw_roll", "-rz+rx", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "ceramic_mug",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_mug_handle",
            conditions={
                "ceramic_mug": [
                    (
                        partial(
                            angled_reach_object,
                            pos_tolerance=0.10,
                            angle_tolerance=0.35,
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
