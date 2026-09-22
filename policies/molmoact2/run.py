"""
Evaluate the MolmoAct2 policy across registered tasks.

"""

from __future__ import annotations

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "molmoact2"

parser = argparse.ArgumentParser(description="Evaluate MolmoAct2-DROID on RoboLab DROID tasks.")

parser.add_argument(
    "--remote-host",
    "--remote_host",
    type=str,
    default="localhost",
    help="MolmoAct2 policy server host.",
)
parser.add_argument(
    "--remote-port",
    "--remote_port",
    type=int,
    default=8000,
    help="MolmoAct2 policy server port.",
)
parser.add_argument(
    "--open-loop-horizon",
    "--open_loop_horizon",
    type=int,
    default=10,
    help="Number of MolmoAct2 actions to consume before requesting a new chunk.",
)
parser.add_argument(
    "--request-timeout",
    "--request_timeout",
    type=float,
    default=60.0,
    help="HTTP request timeout in seconds.",
)
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from policies.molmoact2.client import MolmoAct2Client  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_droid_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    randomize_background=args_cli.randomize_background,
    background_seed=args_cli.background_seed,
)


def make_client(args: argparse.Namespace) -> MolmoAct2Client:
    return MolmoAct2Client(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        open_loop_horizon=args.open_loop_horizon,
        request_timeout=args.request_timeout,
    )


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)