# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demo for running a generated environment configuration.

This script demonstrates how to create an environment from a task file
and run episodes with gripper toggling or policy control.

Usage:
    $ python run_env.py

    Run headless:
    $ python run_env.py --headless
"""

import os
import argparse

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

DEFAULT_KIT_ARGS = "--/app/livestream/publicEndpointAddress=172.29.5.11  --/app/livestream/port=49100"
DEFAULT_VIEWER_EYE = (0.05, 0.57, 0.66)
DEFAULT_VIEWER_LOOKAT = (0.55, 0.19, 0.17)

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo on using the mimic joints for Robotiq 140 gripper.")
parser.add_argument("--num_envs", 
                    type=int, 
                    default=1, 
                    help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# isaac webRTC live streaming settings
args_cli.livestream = 2
args_cli.kit_args = DEFAULT_KIT_ARGS

# enable cameras and video saving
args_cli.enable_cameras = True
args_cli.activate_contact_sensors = True
args_cli.save_videos = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import tyro
from episodes import run_empty_episode, run_gripper_toggle_episode, run_prerecorded_episode

from robolab.constants import DEFAULT_OUTPUT_DIR, PACKAGE_DIR, TASK_DIR, get_timestamp
from robolab.core.environments.config import generate_env_cfg_from_task  # noqa
from robolab.core.environments.runtime import create_env  # noqa
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg  # noqa
from robolab.robots.droid import (  # noqa
    DroidCfg,
    DroidIKActionCfg,
    ProprioceptionObservationCfg,
    WristCameraCfg,
    contact_gripper,
)
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.registrations.droid_jointpos.camera_presets import WRIST_RIGHT
# from robolab.variations.lighting import SphereLightCfg  # noqa


def main():
    """Main function."""

    num_episodes = 2
    env = None

    ImageObsCfg = generate_image_obs_from_cameras(WRIST_RIGHT)
    # WristCameraCfg is already mounted in DroidCfg; keep it in observations
    # but exclude it from scene camera mixins to preserve spawn ordering.
    scene_cameras = [c for c in WRIST_RIGHT if c is not WristCameraCfg]
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ImageObsCfg()
    })

    # # Setup environment
    EnvCfg, _ = generate_env_cfg_from_task(
        task_file_path=f"{TASK_DIR}/wm_tasks/angledreach/angledreach_drill_task.py",
        env_name="DroidAngledReachDrillEnv",
        robot_cfg=DroidCfg,
        camera_cfg=scene_cameras,
        # lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        actions_cfg=DroidIKActionCfg(),
        observations_cfg=ObservationCfg(),
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        eye=DEFAULT_VIEWER_EYE,
        lookat=DEFAULT_VIEWER_LOOKAT,
        env_spacing=2.0,
        num_envs=1,
        seed=0,
    )

    env_cfg = EnvCfg()
    env_cfg.sim.device = args_cli.device
    print(f"Generated environment config")

    # Livestream mode runs without a local desktop window.
    effective_headless = bool(args_cli.headless) or bool(args_cli.livestream)

    try:
        env, _ = create_env(scene=env_cfg,
                         device=args_cli.device,
                         num_envs=args_cli.num_envs,
                         use_fabric=True)
        print("create_env returned", flush=True)

        while simulation_app.is_running():
            simulation_app.update()

    except KeyboardInterrupt:
        print("Ctrl+C received. Closing environment.")

    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    return

if __name__ == "__main__":
    main()
    # args = tyro.cli(main)
