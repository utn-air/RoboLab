# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Goal image cache/generation helpers for VALP world-model tasks.

This module keeps task goal-image capture out of the policy episode runner.
It can also be run directly inside an Isaac Lab Python session to precompute
goal images under ``assets/wm_tasks``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
WM_GOAL_DIR = REPO_ROOT / "assets" / "wm_tasks"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _task_name(env_cfg) -> str:
    return _safe_name(getattr(env_cfg, "_task_name", None) or env_cfg.__class__.__name__)


def _goal_cfg(env_cfg) -> dict:
    goal_cfg = getattr(env_cfg, "valp_goal", None)
    if not goal_cfg:
        raise ValueError(
            f"{_task_name(env_cfg)} does not define valp_goal; cannot generate VALP goal images."
        )
    return dict(goal_cfg)


def goal_image_dir(env_cfg) -> Path:
    return WM_GOAL_DIR / _task_name(env_cfg)


def goal_image_paths(env_cfg) -> dict[str, Path]:
    goal_cfg = _goal_cfg(env_cfg)
    external_key = goal_cfg.get("external_camera", "external_right_cam")
    wrist_key = goal_cfg.get("wrist_camera", "wrist_cam")
    root = goal_image_dir(env_cfg)
    return {
        "external": root / f"{external_key}.png",
        "wrist": root / f"{wrist_key}.png",
        "metadata": root / "metadata.json",
    }


def goal_images_exist(env_cfg) -> bool:
    paths = goal_image_paths(env_cfg)
    return paths["external"].exists() and paths["wrist"].exists()


def _get_action_dim(env, fallback: int = 7) -> int:
    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    shape = getattr(space, "shape", None)
    if shape:
        return int(shape[-1])

    action_manager = getattr(env, "action_manager", None)
    total_dim = getattr(action_manager, "total_action_dim", None)
    return int(total_dim) if total_dim is not None else fallback


def _save_rgb_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = image.detach().cpu().numpy()
    if image_np.dtype != "uint8":
        image_np = image_np.clip(0, 255).astype("uint8")
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_np):
        raise OSError(f"Failed to write goal image: {path}")


def _load_rgb_image(path: Path) -> torch.Tensor:
    image_np = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_np is None:
        raise FileNotFoundError(f"Could not read goal image: {path}")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_np.copy())


def _compute_reach_goal_positions(env, target_object: str, z_offset: float) -> torch.Tensor:
    from robolab.core.world.world_state import get_world

    world = get_world(env)
    corners, centroid = world.get_bbox(target_object, env_id=None)
    target_positions = centroid.clone()
    target_positions[:, 2] = corners[:, :, 2].max(dim=1).values + z_offset
    return target_positions + env.scene.env_origins


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw from wxyz quaternions."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _object_axis_yaw(env, object_name: str, axis: str = "x") -> torch.Tensor:
    """Estimate object yaw from a local axis projected into the world XY plane."""
    from robolab.core.world.world_state import get_world

    _, quat = get_world(env).get_pose(object_name, env_id=None)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    axis = axis.lower()
    if axis == "x":
        axis_world_x = 1.0 - 2.0 * (y * y + z * z)
        axis_world_y = 2.0 * (x * y + w * z)
    elif axis == "y":
        axis_world_x = 2.0 * (x * y - w * z)
        axis_world_y = 1.0 - 2.0 * (x * x + z * z)
    else:
        raise ValueError(f"Unsupported yaw axis '{axis}'. Use 'x' or 'y'.")
    return torch.atan2(axis_world_y, axis_world_x)


def drive_to_valp_goal(env, env_cfg, obs: dict | None = None) -> dict:
    """Drive the robot to the task's configured VALP goal pose and return latest obs."""
    from robolab.core.world.world_state import get_world

    goal_cfg = _goal_cfg(env_cfg)
    mode = goal_cfg.get("mode")
    if mode not in ("reach_above_object", "reach_above_object_with_yaw"):
        raise ValueError(f"Unsupported VALP goal mode: {mode}")

    if obs is None:
        obs, _ = env.reset()

    target_object = goal_cfg["object"]
    z_offset = float(goal_cfg.get("z_offset", 0.12))
    tolerance = float(goal_cfg.get("tolerance", 0.025))
    yaw_tolerance = float(goal_cfg.get("yaw_tolerance", 0.08))
    max_steps = int(goal_cfg.get("drive_steps", 80))
    settle_steps = int(goal_cfg.get("settle_steps", 4))
    ik_action_scale = float(goal_cfg.get("ik_action_scale", 0.5))
    yaw_action_scale = float(goal_cfg.get("yaw_action_scale", 0.5))
    max_action = float(goal_cfg.get("max_action", 0.25))
    max_rot_action = float(goal_cfg.get("max_rot_action", 0.25))
    link_name = goal_cfg.get("link_name", "base_link")
    yaw_action_index = int(goal_cfg.get("yaw_action_index", 5))

    action_dim = _get_action_dim(env)
    target_positions = _compute_reach_goal_positions(env, target_object, z_offset)
    actions = torch.zeros(env.num_envs, action_dim, device=env.device)

    if mode == "reach_above_object_with_yaw":
        yaw_source = goal_cfg.get("yaw_source", "object_axis")
        if yaw_source == "object_axis":
            target_yaw = _object_axis_yaw(env, target_object, goal_cfg.get("object_yaw_axis", "x"))
        elif yaw_source == "constant":
            target_yaw = torch.full(
                (env.num_envs,), float(goal_cfg.get("target_yaw", 0.0)), device=env.device
            )
        else:
            raise ValueError(f"Unsupported yaw_source: {yaw_source}")
        target_yaw = _wrap_to_pi(target_yaw + float(goal_cfg.get("yaw_offset", 0.0)))
    else:
        target_yaw = None

    for _ in range(max_steps):
        gripper_pose = get_world(env).get_articulation_link_pose("robot", link_name, env_id=None)
        pos_error = target_positions - gripper_pose[:, :3]
        pos_done = torch.linalg.norm(pos_error, dim=1).max().item() <= tolerance

        actions.zero_()
        actions[:, :3] = torch.clamp(pos_error / max(ik_action_scale, 1e-6), -max_action, max_action)

        yaw_done = True
        if target_yaw is not None and yaw_action_index < action_dim:
            gripper_yaw = _quat_to_yaw(gripper_pose[:, 3:7])
            yaw_error = _wrap_to_pi(target_yaw - gripper_yaw)
            yaw_done = yaw_error.abs().max().item() <= yaw_tolerance
            actions[:, yaw_action_index] = torch.clamp(
                yaw_error / max(yaw_action_scale, 1e-6), -max_rot_action, max_rot_action
            )

        if pos_done and yaw_done:
            break

        obs, _, _, _, _ = env.step(actions)

    actions.zero_()
    for _ in range(settle_steps):
        obs, _, _, _, _ = env.step(actions)

    return obs


def generate_goal_images(env, env_cfg, obs: dict | None = None, overwrite: bool = False) -> dict[str, Path]:
    """Generate and cache one canonical pair of goal images for a WM task."""
    paths = goal_image_paths(env_cfg)
    if goal_images_exist(env_cfg) and not overwrite:
        return paths

    goal_cfg = _goal_cfg(env_cfg)
    print(f"\033[96m[RoboLab] Generating VALP goal images for {_task_name(env_cfg)}\033[0m")
    goal_obs = drive_to_valp_goal(env, env_cfg, obs=obs)

    external_key = goal_cfg.get("external_camera", "external_right_cam")
    wrist_key = goal_cfg.get("wrist_camera", "wrist_cam")
    _save_rgb_image(goal_obs["image_obs"][external_key][0], paths["external"])
    _save_rgb_image(goal_obs["image_obs"][wrist_key][0], paths["wrist"])

    metadata = {
        "task": _task_name(env_cfg),
        "instruction": getattr(env_cfg, "instruction", None),
        "goal": goal_cfg,
        "external_image": paths["external"].name,
        "wrist_image": paths["wrist"].name,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def ensure_goal_images(env, env_cfg, obs: dict | None = None) -> dict[str, Path]:
    paths = goal_image_paths(env_cfg)
    if not goal_images_exist(env_cfg):
        paths = generate_goal_images(env, env_cfg, obs=obs)
        env.reset_eval_state()
    return paths


def set_client_goal_images(client, env, env_cfg, obs: dict | None, instruction: str) -> dict:
    """Ensure cached images exist, load them, set them on the VALP client, and reset env."""
    paths = ensure_goal_images(env, env_cfg, obs=obs)
    external_goal = _load_rgb_image(paths["external"])
    wrist_goal = _load_rgb_image(paths["wrist"])

    for env_id in range(env.num_envs):
        client.set_goal_images(external_goal, wrist_goal, env_id=env_id, instruction=instruction)

    env.reset_eval_state()
    obs, _ = env.reset()
    return obs


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Generate cached VALP goal images for WM tasks.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--task", required=True, help="Task name to generate, e.g. ReachBananaTask.")
    parser.add_argument("--task-dirs", nargs="+", default=["wm_tasks"], help="Task folders to register.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--instruction-type", "--instruction_type", default="default")
    parser.add_argument("--overwrite", action="store_true")
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid_ee.auto_env_registrations import auto_register_droid_ee_envs

        auto_register_droid_ee_envs(task_dirs=args_cli.task_dirs, task=[args_cli.task])
        task_envs = get_envs(task=[args_cli.task])
        if not task_envs:
            raise ValueError(f"Task '{args_cli.task}' was not registered.")

        env, env_cfg = create_env(
            task_envs[0],
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy="valp",
        )
        generate_goal_images(env, env_cfg, overwrite=args_cli.overwrite)
        env.close()
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
