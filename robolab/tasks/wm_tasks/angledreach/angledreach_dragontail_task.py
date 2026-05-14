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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachDragontailTask" / "status.json"


@configclass
class AngledReachDragontailTerminations:
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
class AngledReachDragontailTask(Task):
    contact_object_list = [
        "table",
        "glasses",
        "lizard_figurine",
        "marker",
        "remote_control",
        "rubiks_cube",
        "smartphone",
    ]
    scene = import_scene("workdesk.usda", contact_object_list)
    terminations = AngledReachDragontailTerminations
    instruction = {
        "default": "AngledReachDragontail",
        "vague": "Reach the left edge of the dragontail with a rolled wrist",
        "specific": "Move the robot gripper to the left edge of the dragontail near the center of the table with the wrist rolled so the fingers align vertically with the thin dragontail side, without grasping it",
    }
    episode_steps: int = 100
    attributes = ["angled_reach", "dominant_roll", "-rx", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "lizard_figurine",
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_dragontail",
            conditions={
                "lizard_figurine": [
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
