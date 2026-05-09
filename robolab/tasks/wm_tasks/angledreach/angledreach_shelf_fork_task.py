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

STATUS_PATH = Path(ASSET_DIR) / "wm_tasks" / "AngledReachShelfForkTask" / "status.json"

@configclass
class AngledReachShelfForkTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=angled_reach_object,
        params={"object": "fork_big", "tolerance": 0.06, "status_path": STATUS_PATH},
    )


@dataclass
class AngledReachShelfForkTask(Task):
    contact_object_list = [
        "table",
        "sm_rack_m01",
        "spatula_05",
        "spoon_big",
        "spoon_small",
        "fork_big",
        "fork_small",
        "milkjug_a01",
        "blackandbrassbowl_large",
        "gardenplanter_large",
    ]
    scene = import_scene("front_of_shelf.usda", contact_object_list)
    terminations = AngledReachShelfForkTerminations
    instruction = {
        "default": "AngledReachShelfFork",
        "vague": "Reach the fork on the second shelf with pitched approach",
        "specific": "Move the robot gripper to the fork on the second shelf facing the handle without grasping it",
    }
    episode_steps: int = 50
    attributes = ["angled_reach", "goal"]
    goal = {
        "mode": "angled_reach",
        "object": "fork_big",
        "drive_steps": 30,
        "settle_steps": 4,
        "external_camera": "over_shoulder_right_camera",
        "wrist_camera": "wrist_cam",
    }
    subtasks = [
        Subtask(
            name="angled_reach_fork_big",
            conditions={
                "fork_big": [
                    (partial(angled_reach_object, object="fork_big", tolerance=0.06, status_path=STATUS_PATH), 1.0)
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
