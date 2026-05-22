# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard-driven pickup goal capture for generated DROID WM environments.

This script resumes from the first angled-reach goal stored in:

    assets/wm_tasks/<TaskName>/status.json
    assets/wm_tasks/<TaskName>/over_shoulder_right_camera_1.png
    assets/wm_tasks/<TaskName>/wrist_cam_1.png

It drives the robot to ``last_ee_pose`` or ``last_ee_pose_1`` when available,
then keeps the same terminal jog controls as run_env_keys.py. Press ``g`` to
close the gripper, save goal 2, lift the held object toward a home-safe pose
30 cm above the table, and save goal 3.

Usage:
    $ python examples/demo/run_env_keys_pickup_goals.py --headless --task AngledPickupDrillTask
"""

import argparse
import json
import math
import re
from pathlib import Path

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

DEFAULT_KIT_ARGS = "--/app/livestream/publicEndpointAddress=172.29.5.11  --/app/livestream/port=49100"

parser = argparse.ArgumentParser(description="Keyboard EE jogger for pickup WM goal capture.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn.",
)
parser.add_argument(
    "--task",
    type=str,
    default="AngledPickupDrillTask",
    help="Registered pickup task to open.",
)
parser.add_argument(
    "--task-dirs",
    nargs="+",
    default=["wm_tasks/andgedpickup"],
    help="Task folders passed to the angled EE registrar.",
)
parser.add_argument(
    "--pos-step",
    type=float,
    default=0.15,
    help="Translation jog size in meters before the IK action scale is applied.",
)
parser.add_argument(
    "--rot-step",
    type=float,
    default=0.75,
    help="Angle-axis rotation jog size in radians before the IK action scale is applied.",
)
parser.add_argument(
    "--repeat-steps",
    type=int,
    default=1,
    help="Number of env steps to repeat each nonzero key command.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=0,
    help="Zero-action steps after each key command before saving images.",
)
parser.add_argument(
    "--drive-steps",
    type=int,
    default=120,
    help="Maximum steps for automated pose drives.",
)
parser.add_argument(
    "--drive-pos-tol",
    type=float,
    default=0.01,
    help="Position tolerance in meters for automated pose drives.",
)
parser.add_argument(
    "--drive-angle-tol",
    type=float,
    default=0.08,
    help="Orientation tolerance in radians for automated pose drives.",
)
parser.add_argument(
    "--max-drive-action",
    type=float,
    default=0.10,
    help="Maximum absolute xyz/rotation action component during automated drives.",
)
parser.add_argument(
    "--gripper-steps",
    type=int,
    default=12,
    help="Number of repeated gripper steps for open/close commands.",
)
parser.add_argument(
    "--lift-height",
    type=float,
    default=0.30,
    help="Goal-3 clearance above the table top in meters.",
)
parser.add_argument(
    "--force-livestream",
    action="store_true",
    help="Force WebRTC livestream mode in addition to terminal control.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.force_livestream:
    args_cli.livestream = 2
    args_cli.kit_args = DEFAULT_KIT_ARGS

args_cli.enable_cameras = True
args_cli.activate_contact_sensors = True
args_cli.save_videos = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.utils.math as math_utils  # noqa

from robolab.core.environments.runtime import create_env  # noqa
from robolab.core.world.world_state import get_world
from robolab.tasks.wm_tasks.goal_images import _save_rgb_image, goal_image_paths


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

SIMPLE_HELP = """
Type keys, then Enter. Repeated letters repeat the move.
  w/s: +x/-x    a/d: +y/-y    r/f: +z/-z
  i/k: +rx/-rx  j/l: +ry/-ry  u/o: +rz/-rz  (angle-axis)
  c/v: close/open gripper
  .: save current as goal 1 draft
  g: close, save goal 2, lift 30 cm above table, save goal 3
  x: reset and redrive to goal 1      q: quit
"""


def _zero_action(env):
    return torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)


def _simple_action(env, key: str):
    action = _zero_action(env)
    commands = {
        "w": (0, args_cli.pos_step),
        "s": (0, -args_cli.pos_step),
        "a": (1, args_cli.pos_step),
        "d": (1, -args_cli.pos_step),
        "r": (2, args_cli.pos_step),
        "f": (2, -args_cli.pos_step),
        "i": (3, args_cli.rot_step),
        "k": (3, -args_cli.rot_step),
        "j": (4, args_cli.rot_step),
        "l": (4, -args_cli.rot_step),
        "u": (5, args_cli.rot_step),
        "o": (5, -args_cli.rot_step),
        "c": (6, 1.0),
        "v": (6, -1.0),
    }
    if key not in commands:
        return None
    idx, value = commands[key]
    action[:, idx] = value
    return action


def _clean_keys(keys: str) -> str:
    keys = ANSI_ESCAPE_RE.sub("", keys)
    return "".join(key for key in keys.lower() if key in "wsadrfijkluocv.xqgh")


def _task_goal_root(env_cfg) -> Path:
    paths = goal_image_paths(env_cfg)
    return paths["status"].parent


def _load_status(env_cfg) -> dict:
    status_path = _task_goal_root(env_cfg) / "status.json"
    if not status_path.exists():
        print(f"no status.json found at {status_path}; starting from reset pose", flush=True)
        return {}
    with status_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_goal1_pose(env_cfg):
    status = _load_status(env_cfg)
    pose = status.get("last_ee_pose_1", status.get("last_ee_pose"))
    if pose is None:
        return None
    if len(pose) != 7:
        raise ValueError(f"Expected a 7D ee pose in status.json, got {len(pose)} values")
    return pose


def _read_status_for_update(env_cfg) -> dict:
    status_path = _task_goal_root(env_cfg) / "status.json"
    if not status_path.exists():
        return {}
    with status_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_status_entry(env_cfg, suffix: int | None, ee_pose: list[float]) -> None:
    status_path = _task_goal_root(env_cfg) / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_status_for_update(env_cfg)
    payload["reached"] = True
    payload["manual_capture"] = True
    payload["last_distance"] = 0.0
    if suffix is None:
        payload["last_ee_pose"] = ee_pose
        payload.setdefault("last_ee_pose_1", ee_pose)
    else:
        payload[f"last_ee_pose_{suffix}"] = ee_pose
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_goal(env, env_cfg, obs, suffix: int | None, label: str):
    paths = goal_image_paths(env_cfg)
    external_key = env_cfg.goal.get("external_camera", "over_shoulder_right_camera")
    wrist_key = env_cfg.goal.get("wrist_camera", "wrist_cam")
    image_obs = obs["image_obs"]

    external_path = paths["external"]
    wrist_path = paths["wrist"]
    if suffix is not None:
        external_path = external_path.with_name(f"{external_path.stem}_{suffix}{external_path.suffix}")
        wrist_path = wrist_path.with_name(f"{wrist_path.stem}_{suffix}{wrist_path.suffix}")

    _save_rgb_image(image_obs[external_key][0], external_path)
    _save_rgb_image(image_obs[wrist_key][0], wrist_path)

    link_name = env_cfg.goal.get("link_name", "panda_link8")
    ee_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)[0]
    ee_pose = [float(x) for x in ee_pose.detach().cpu().tolist()]
    _write_status_entry(env_cfg, suffix, ee_pose)
    print(f"saved {label}: {external_path.name}, {wrist_path.name}", flush=True)
    return obs


def _quat_error_axis_angle(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    current_quat = torch.nn.functional.normalize(current_quat, dim=-1)
    target_quat = torch.nn.functional.normalize(target_quat, dim=-1)

    same_side_target = torch.where(
        (current_quat * target_quat).sum(dim=-1, keepdim=True) < 0.0,
        -target_quat,
        target_quat,
    )
    delta_quat = math_utils.quat_mul(same_side_target, math_utils.quat_conjugate(current_quat))
    return math_utils.axis_angle_from_quat(delta_quat)


def _drive_to_pose(env, env_cfg, obs, target_pose, *, gripper_action=0.0, label="target"):
    link_name = env_cfg.goal.get("link_name", "panda_link8")
    target = torch.tensor(target_pose, dtype=torch.float32, device=env.device)
    target_pos = target[:3].unsqueeze(0).repeat(env.num_envs, 1)
    target_quat = target[3:7].unsqueeze(0).repeat(env.num_envs, 1)
    reached = False
    last_pos_err = math.inf
    last_angle_err = math.inf

    for _ in range(max(1, args_cli.drive_steps)):
        current_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)
        pos_error = target_pos - current_pose[:, :3]
        rot_error = _quat_error_axis_angle(current_pose[:, 3:7], target_quat)

        last_pos_err = torch.linalg.norm(pos_error, dim=1).max().item()
        last_angle_err = torch.linalg.norm(rot_error, dim=1).max().item()
        if last_pos_err <= args_cli.drive_pos_tol and last_angle_err <= args_cli.drive_angle_tol:
            reached = True
            break

        action = _zero_action(env)
        action[:, :3] = torch.clamp(pos_error, -args_cli.max_drive_action, args_cli.max_drive_action)
        action[:, 3:6] = torch.clamp(rot_error, -args_cli.max_drive_action, args_cli.max_drive_action)
        action[:, 6] = gripper_action
        obs, _, _, _, _ = env.step(action)

    for _ in range(max(0, args_cli.settle_steps)):
        obs, _, _, _, _ = env.step(_zero_action(env))

    status = "reached" if reached else "stopped"
    print(
        f"drive {status} {label}: pos_err={last_pos_err:.4f}, angle_err={last_angle_err:.4f}",
        flush=True,
    )
    return obs


def _apply_gripper(env, obs, value: float, label: str):
    action = _zero_action(env)
    action[:, 6] = value
    for _ in range(max(1, args_cli.gripper_steps)):
        obs, _, _, _, _ = env.step(action)
    for _ in range(max(0, args_cli.settle_steps)):
        obs, _, _, _, _ = env.step(_zero_action(env))
    print(label, flush=True)
    return obs


def _current_ee_pose(env, env_cfg) -> torch.Tensor:
    link_name = env_cfg.goal.get("link_name", "panda_link8")
    return get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)[0]


def _table_clearance_pose(env, env_cfg, clearance: float, home_pose: torch.Tensor | None) -> list[float]:
    current_pose = _current_ee_pose(env, env_cfg)
    target_pose = current_pose.detach().clone()
    if home_pose is not None:
        target_pose[:2] = home_pose[:2].to(device=target_pose.device, dtype=target_pose.dtype)

    world = get_world(env)
    corners, _ = world.get_bbox("table", env_id=None)
    table_top_z = corners[:, :, 2].max(dim=1).values[0]
    target_pose[2] = table_top_z + clearance

    return [float(x) for x in target_pose.detach().cpu().tolist()]


def _capture_pickup_sequence(env, env_cfg, obs, home_pose):
    obs = _apply_gripper(env, obs, 1.0, "closed gripper for goal 2")
    obs = _save_goal(env, env_cfg, obs, 2, "goal 2 grasp")

    lift_pose = _table_clearance_pose(env, env_cfg, args_cli.lift_height, home_pose)
    obs = _drive_to_pose(
        env,
        env_cfg,
        obs,
        lift_pose,
        gripper_action=1.0,
        label=f"goal 3 lift ({args_cli.lift_height:.2f}m above table)",
    )
    obs = _save_goal(env, env_cfg, obs, 3, "goal 3 lifted")
    return obs


def main():
    from robolab.core.environments.factory import get_envs
    from robolab.registrations.droid_ee.auto_env_registrations_angled import auto_register_droid_ee_envs

    env = None
    try:
        auto_register_droid_ee_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)
        task_envs = get_envs(task=args_cli.task)
        if not task_envs:
            raise RuntimeError(f"No registered env found for task {args_cli.task!r}")

        env, env_cfg = create_env(
            task_envs[0],
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            policy="valp",
        )
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(_zero_action(env))
        home_pose = _current_ee_pose(env, env_cfg).detach().clone()

        goal1_pose = _load_goal1_pose(env_cfg)
        if goal1_pose is not None:
            obs = _drive_to_pose(env, env_cfg, obs, goal1_pose, label="goal 1 from status.json")

        print(SIMPLE_HELP, flush=True)

        while True:
            keys = _clean_keys(input("key> ").strip())
            if not keys:
                continue
            if "q" in keys:
                break
            if "h" in keys:
                print(SIMPLE_HELP, flush=True)
                continue
            if "x" in keys:
                obs, _ = env.reset()
                obs, _, _, _, _ = env.step(_zero_action(env))
                home_pose = _current_ee_pose(env, env_cfg).detach().clone()
                goal1_pose = _load_goal1_pose(env_cfg)
                if goal1_pose is not None:
                    obs = _drive_to_pose(env, env_cfg, obs, goal1_pose, label="goal 1 from status.json")
                continue

            applied = 0
            for key in keys:
                if key == ".":
                    obs = _save_goal(env, env_cfg, obs, 1, "goal 1 draft")
                    continue
                if key == "g":
                    obs = _capture_pickup_sequence(env, env_cfg, obs, home_pose)
                    continue

                action = _simple_action(env, key)
                if action is None:
                    print(f"unknown key {key!r}; skipping", flush=True)
                    continue

                repeat_steps = args_cli.gripper_steps if key in "cv" else args_cli.repeat_steps
                for _ in range(max(1, repeat_steps)):
                    obs, _, _, _, _ = env.step(action)
                applied += 1

            if applied == 0:
                continue
            for _ in range(max(0, args_cli.settle_steps)):
                obs, _, _, _, _ = env.step(_zero_action(env))

    except KeyboardInterrupt:
        print("\nCtrl+C received. Closing environment.", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
