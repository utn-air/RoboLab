"""Evaluate DUET-DINO using successive goals from a recorded RoboLab video."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401 -- Isaac Lab requires OpenCV to be imported first.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--remote-host", default="localhost")
parser.add_argument("--remote-port", type=int, default=8000)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
parser.set_defaults(task_dirs=["my_custom_tasks"])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if not args_cli.task or len(args_cli.task) != 1:
    parser.error("DUET-DINO requires exactly one --task")
args_cli.enable_cameras = True
args_cli.num_envs = 1
args_cli.num_runs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.core.task.task_utils import find_task_files, load_task_from_file  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_rel_ik import (  # noqa: E402
    auto_register_droid_rel_ik_envs,
)
from policies.duet_dino.client import DUETDINOClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
auto_register_droid_rel_ik_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)


def make_client(args: argparse.Namespace) -> DUETDINOClient:
    task_classes = [
        task_class
        for task_file in find_task_efiles(robolab.constants.TASK_DIR, subfolders=args.task_dirs)
        for task_class in load_task_from_file(task_file, allow_multiple=True)
        if task_class.__name__ == args.task[0]
    ]
    if len(task_classes) != 1:
        raise ValueError(f"Expected one task definition for {args.task[0]}, found {len(task_classes)}")
    task = task_classes[0]
    task_dir = Path(task.task_dir)
    if not task_dir.is_absolute():
        task_dir = Path(robolab.constants.PACKAGE_DIR) / task_dir
    return DUETDINOClient(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        task_dir=task_dir,
        run=task.run,
        key_frames=task.key_frames,
        planning_steps=task.planning_steps,
    )


if __name__ == "__main__":
    try:
        run_evaluation(args_cli, policy="duet_dino", client_factory=make_client)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
