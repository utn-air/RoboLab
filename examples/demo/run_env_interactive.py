# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demo for running a generated environment configuration.

This script demonstrates how to create an environment from a task file
and manually capture goal images from sensor cameras.

Usage:
    $ python examples/demo/run_env.py

    Capture a different task:
    $ python examples/demo/run_env.py \
        --task-file robolab/tasks/wm_tasks/angledreach/angledreach_ketchup_task.py
"""

import argparse
import json
from pathlib import Path

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
parser.add_argument(
    "--task-file",
    type=str,
    default="robolab/tasks/wm_tasks/angledreach/angledreach_drill_task.py",
    help="Task file to open for manual goal capture.",
)
parser.add_argument(
    "--env-name",
    type=str,
    default="DroidManualGoalCaptureEnv",
    help="Temporary environment name used for this generated task.",
)
parser.add_argument(
    "--control-mode",
    choices=["ik", "joints"],
    default="ik",
    help="Use end-effector jog sliders or absolute joint-position sliders.",
)
parser.add_argument(
    "--no-livestream",
    action="store_true",
    help="Do not force WebRTC livestream mode.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# isaac webRTC live streaming settings
if not args_cli.no_livestream:
    args_cli.livestream = 2
    args_cli.kit_args = DEFAULT_KIT_ARGS

# enable cameras and video saving
args_cli.enable_cameras = True
args_cli.activate_contact_sensors = True
args_cli.save_videos = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from robolab.constants import TASK_DIR
from robolab.core.environments.config import generate_env_cfg_from_task  # noqa
from robolab.core.environments.runtime import create_env  # noqa
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg  # noqa
from robolab.robots.droid import (  # noqa
    DroidCfg,
    DroidIKActionCfg,
    DroidJointPositionActionCfg,
    ProprioceptionObservationCfg,
    WristCameraCfg,
    contact_gripper,
)
from robolab.core.world.world_state import get_world
from robolab.tasks.wm_tasks.goal_images import _save_rgb_image, goal_image_paths
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.registrations.droid_jointpos.camera_presets import WRIST_RIGHT
# from robolab.variations.lighting import SphereLightCfg  # noqa


def _resolve_task_file(task_file: str) -> str:
    path = Path(task_file).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)

    task_dir_path = Path(TASK_DIR) / path
    return str(task_dir_path)


class ManualGoalCapturePanel:
    """Small Isaac UI panel for mouse-driven robot jogging and goal capture."""

    def __init__(self, env, env_cfg, control_mode: str):
        self.env = env
        self.env_cfg = env_cfg
        self.control_mode = control_mode
        self.last_obs = None
        self.capture_requested = False
        self.reset_requested = False
        self.ik_command = [0.0] * 7
        self.joint_targets = self._current_joint_targets()
        self.gripper_closed = 0.0
        self._build_ui()

    def _current_joint_targets(self) -> list[float]:
        robot = self.env.scene["robot"]
        return robot.data.joint_pos[0, :7].detach().cpu().tolist()

    def _build_ui(self) -> None:
        import omni.ui as ui

        self.window = ui.Window("Manual Goal Capture", width=380, height=560)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label(f"Task: {getattr(self.env_cfg, '_task_name', self.env_cfg.__class__.__name__)}")
                ui.Label(f"Mode: {self.control_mode}")

                if self.control_mode == "ik":
                    self._build_ik_controls(ui)
                else:
                    self._build_joint_controls(ui)

                with ui.HStack(height=32, spacing=8):
                    ui.Button("Capture goal", clicked_fn=self._request_capture)
                    ui.Button("Reset env", clicked_fn=self._request_reset)

                self.status_label = ui.Label("Drag sliders, then click Capture goal.")

    def _build_ik_controls(self, ui) -> None:
        labels = [
            ("dx", -0.04, 0.04),
            ("dy", -0.04, 0.04),
            ("dz", -0.04, 0.04),
            ("droll", -0.12, 0.12),
            ("dpitch", -0.12, 0.12),
            ("dyaw", -0.12, 0.12),
            ("grip", -0.05, 0.05),
        ]
        self.ik_models = []
        for idx, (label, min_value, max_value) in enumerate(labels):
            with ui.HStack(height=26, spacing=8):
                ui.Label(label, width=58)
                slider = ui.FloatSlider(min=min_value, max=max_value)
                slider.model.set_value(0.0)
                slider.model.add_value_changed_fn(self._ik_callback(idx))
                self.ik_models.append(slider.model)

        ui.Button("Zero jog sliders", height=28, clicked_fn=self._zero_ik_controls)

    def _build_joint_controls(self, ui) -> None:
        self.joint_models = []
        for idx in range(7):
            with ui.HStack(height=26, spacing=8):
                ui.Label(f"joint{idx + 1}", width=58)
                slider = ui.FloatSlider(min=-3.14, max=3.14)
                slider.model.set_value(float(self.joint_targets[idx]))
                slider.model.add_value_changed_fn(self._joint_callback(idx))
                self.joint_models.append(slider.model)

        with ui.HStack(height=26, spacing=8):
            ui.Label("gripper", width=58)
            slider = ui.FloatSlider(min=0.0, max=1.0)
            slider.model.set_value(0.0)
            slider.model.add_value_changed_fn(self._gripper_callback)
            self.gripper_model = slider.model

    def _ik_callback(self, idx: int):
        def callback(model):
            self.ik_command[idx] = float(model.get_value_as_float())
        return callback

    def _joint_callback(self, idx: int):
        def callback(model):
            self.joint_targets[idx] = float(model.get_value_as_float())
        return callback

    def _gripper_callback(self, model) -> None:
        self.gripper_closed = float(model.get_value_as_float())

    def _zero_ik_controls(self) -> None:
        self.ik_command = [0.0] * 7
        for model in self.ik_models:
            model.set_value(0.0)

    def _request_capture(self) -> None:
        self.capture_requested = True

    def _request_reset(self) -> None:
        self.reset_requested = True

    def action(self) -> torch.Tensor:
        if self.control_mode == "ik":
            action = torch.tensor(self.ik_command, dtype=torch.float32, device=self.env.device)
        else:
            command = self.joint_targets + [self.gripper_closed]
            action = torch.tensor(command, dtype=torch.float32, device=self.env.device)
        return action.unsqueeze(0).repeat(self.env.num_envs, 1)

    def handle_reset_if_requested(self):
        if not self.reset_requested:
            return None
        self.reset_requested = False
        obs, _ = self.env.reset()
        self.last_obs = obs
        self.joint_targets = self._current_joint_targets()
        if self.control_mode == "joints":
            for idx, model in enumerate(self.joint_models):
                model.set_value(float(self.joint_targets[idx]))
        self.status_label.text = "Environment reset."
        return obs

    def handle_capture_if_requested(self) -> None:
        if not self.capture_requested:
            return
        self.capture_requested = False
        if self.last_obs is None:
            self.status_label.text = "No observation yet; step once before capturing."
            return

        paths = goal_image_paths(self.env_cfg)
        external_key = self.env_cfg.goal.get("external_camera", "over_shoulder_right_camera")
        wrist_key = self.env_cfg.goal.get("wrist_camera", "wrist_cam")
        image_obs = self.last_obs.get("image_obs", {})

        missing = [key for key in (external_key, wrist_key) if key not in image_obs]
        if missing:
            self.status_label.text = f"Missing camera obs: {missing}"
            print(f"[ManualGoalCapture] Missing camera obs: {missing}")
            return

        _save_rgb_image(image_obs[external_key][0], paths["external"])
        _save_rgb_image(image_obs[wrist_key][0], paths["wrist"])

        link_name = self.env_cfg.goal.get("link_name", "panda_link8")
        ee_pose = get_world(self.env).get_articulation_link_pose("robot", link_name, env_id=None)[0]
        ee_pose = [float(x) for x in ee_pose.detach().cpu().tolist()]
        status_payload = {
            "reached": True,
            "manual_capture": True,
            "last_distance": 0.0,
            "last_ee_pose": ee_pose,
        }
        paths["status"].parent.mkdir(parents=True, exist_ok=True)
        with paths["status"].open("w", encoding="utf-8") as handle:
            json.dump(status_payload, handle, indent=2)

        self.status_label.text = f"Saved goal images to {paths['status'].parent}"
        print(f"[ManualGoalCapture] Saved external image: {paths['external']}")
        print(f"[ManualGoalCapture] Saved wrist image: {paths['wrist']}")
        print(f"[ManualGoalCapture] Saved status: {paths['status']}")


def main():
    """Main function."""

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

    task_file_path = _resolve_task_file(args_cli.task_file)
    actions_cfg = DroidIKActionCfg() if args_cli.control_mode == "ik" else DroidJointPositionActionCfg()

    # # Setup environment
    EnvCfg, _ = generate_env_cfg_from_task(
        task_file_path=task_file_path,
        env_name=args_cli.env_name,
        robot_cfg=DroidCfg,
        camera_cfg=scene_cameras,
        # lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        actions_cfg=actions_cfg,
        observations_cfg=ObservationCfg(),
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        eye=DEFAULT_VIEWER_EYE,
        lookat=DEFAULT_VIEWER_LOOKAT,
        env_spacing=2.0,
        num_envs=args_cli.num_envs,
        seed=0,
    )

    env_cfg = EnvCfg()
    env_cfg.sim.device = args_cli.device
    print(f"Generated environment config")

    try:
        env, _ = create_env(scene=env_cfg,
                         device=args_cli.device,
                         num_envs=args_cli.num_envs,
                         use_fabric=True)
        print("create_env returned", flush=True)

        obs, _ = env.reset()
        panel = ManualGoalCapturePanel(env, env_cfg, args_cli.control_mode)
        panel.last_obs = obs

        while simulation_app.is_running():
            reset_obs = panel.handle_reset_if_requested()
            if reset_obs is not None:
                obs = reset_obs
            obs, _, _, _, _ = env.step(panel.action())
            panel.last_obs = obs
            panel.handle_capture_if_requested()

    except KeyboardInterrupt:
        print("Ctrl+C received. Closing environment.")

    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    return

if __name__ == "__main__":
    main()
