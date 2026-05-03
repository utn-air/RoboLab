from __future__ import annotations

import os
import pickle
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .base_client import InferenceClient


REPO_ROOT = Path(__file__).resolve().parents[2]
VALP_ROOT = REPO_ROOT / "valp"

if str(VALP_ROOT) not in sys.path:
    sys.path.insert(0, str(VALP_ROOT))


################################ CLIENT ############################################

class VALPDroidEEClient(InferenceClient):
    """Local RoboLab inference client for the VALP world model on DroidIK envs."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
    ) -> None:
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.sock = self._connect()

    def _connect(self) -> socket.socket:
        print(f"[{self.__class__.__name__}] Waiting for VALP server on {self.remote_host}:{self.remote_port}...")
        while True:
            try:
                sock = socket.create_connection((self.remote_host, self.remote_port), timeout=10)
                sock.settimeout(None)
                print(f"[{self.__class__.__name__}] Connected to VALP server.")
                return sock
            except OSError:
                time.sleep(2)

    def _request(self, payload: dict) -> dict:
        _send_message(self.sock, payload)
        response = _recv_message(self.sock)
        if error := response.get("error"):
            raise RuntimeError(f"VALP server error:\n{error}")
        return response

    def reset(self):
        self._request({"method": "reset"})

    def set_goal_images(self, external_image, wrist_image, *, env_id: int = 0, instruction: str = "goal"):
        # SEND EXACT PARAMS PASSED TO THIS FUNCTION TO SERVER
        self._request(
            {
                "method": "set_goal_images",
                "external_image": external_image,
                "wrist_image": wrist_image,
                "env_id": env_id,
                "instruction": instruction
            }
        )

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        # SEND EXACT PARAMS PASSED TO THIS FUNCTION TO SERVER 
        
        return self._request(
            {
                "method": "infer",
                "obs": obs,
                "instruction": instruction,
                "env_id": env_id,
            }
        )

MyPolicyClient = VALPDroidEEClient

########### SERVER ################
class VALPDroidEEPolicy:
    """Local RoboLab inference client for the VALP world model on DroidIK envs."""

    def __init__(
        self,
        cfg_path: str | None = None,
    ) -> None:
        import yaml

        self.cfg_path = Path.joinpath(VALP_ROOT, cfg_path)
        with self.cfg_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = os.environ.get("VALP_DEVICE", self.cfg.get("device", "cuda:0"))
        self.rollout_horizon = 1
        self._env_prev_action: dict[int, object] = {}
        self._env_goal_rep: dict[int, object] = {}
        self._env_goal_rep_wrist: dict[int, object] = {}

        self._initialize_model()

    def _initialize_model(self) -> None:
        import copy

        import torch

        from app.vjepa_rig.transforms import make_transforms
        from app.vjepa_rig.utils import init_video_model, load_checkpoint, load_pretrained
        from inference.utils.world_model_wrapper import WorldModel

        cfgs_model = self.cfg.get("model", {})
        cfgs_meta = self.cfg.get("meta", {})
        cfgs_data = self.cfg.get("data", {})
        cfgs_data_aug = self.cfg.get("data_aug", {})
        cfgs_mpc_args = self.cfg.get("mpc_args", {})
        cfgs_log_args = self.cfg.get("log", {})
        cfgs_exp_args = self.cfg.get("exp", {})

        camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])
        crop_size = cfgs_data.get("crop_size", 256)
        patch_size = cfgs_data.get("patch_size")
        tubelet_size = cfgs_data.get("tubelet_size")

        transform = make_transforms(
            random_horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
            random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
            random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]),
            reprob=cfgs_data_aug.get("reprob", 0.0),
            auto_augment=cfgs_data_aug.get("auto_augment", False),
            motion_shift=cfgs_data_aug.get("motion_shift", False),
            crop_size=crop_size,
        )

        model_name = cfgs_model.get("model_name")
        pred_depth = cfgs_model.get("pred_depth")
        pred_num_heads = cfgs_model.get("pred_num_heads", None)
        cross_attn_num_heads = cfgs_model.get("cross_attn_num_heads", None)
        pred_embed_dim = cfgs_model.get("pred_embed_dim")
        pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
        uniform_power = cfgs_model.get("uniform_power", False)
        use_rope = cfgs_model.get("use_rope", False)
        use_silu = cfgs_model.get("use_silu", False)
        use_pred_silu = cfgs_model.get("use_pred_silu", False)
        wide_silu = cfgs_model.get("wide_silu", True)
        use_extrinsics = cfgs_model.get("use_extrinsics", False)
        dual_view_training = bool(cfgs_model.get("dual_view_training", False))
        use_dinov3_encoder = bool(cfgs_model.get("use_dinov3_encoder", False))
        use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
        use_sdpa = bool(cfgs_meta.get("use_sdpa", False))

        context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
        pretrain_checkpoint = str(REPO_ROOT / cfgs_meta.get("pretrain_checkpoint"))
        predictor_checkpoint = cfgs_model.get("predictor_checkpoint")
        pretrain_dinocheckpoint = str(REPO_ROOT / cfgs_meta.get("pretrain_dinocheckpoint"))
        if isinstance(predictor_checkpoint, list):
            predictor_checkpoint = [str(REPO_ROOT / ckpt) for ckpt in predictor_checkpoint]
        else:
            predictor_checkpoint = str(REPO_ROOT / predictor_checkpoint)

        if dual_view_training:
            inferred_mode = "dual"
        elif len(camera_views) == 1:
            inferred_mode = "wrist" if "wrist" in camera_views[0] else "side"
        elif len(camera_views) == 2:
            inferred_mode = "independent_dual"
        else:
            raise ValueError(f"Unsupported VALP camera_views configuration: {camera_views}")

        encoder, predictor = init_video_model(
            uniform_power=uniform_power,
            device=self.device,
            patch_size=patch_size,
            max_num_frames=512,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_num_heads=pred_num_heads,
            pred_embed_dim=pred_embed_dim,
            action_embed_dim=7,
            pred_is_frame_causal=pred_is_frame_causal,
            use_extrinsics=use_extrinsics,
            use_sdpa=use_sdpa,
            use_silu=use_silu,
            use_pred_silu=use_pred_silu,
            wide_silu=wide_silu,
            use_rope=use_rope,
            use_activation_checkpointing=use_activation_checkpointing,
            dual_view_predictor=dual_view_training,
            cross_attn_num_heads=cross_attn_num_heads,
            use_dinov3_encoder=use_dinov3_encoder,
        )

        encoder = load_pretrained(
            r_path=pretrain_dinocheckpoint if use_dinov3_encoder else pretrain_checkpoint,
            target_encoder=encoder,
            context_encoder_key=context_encoder_key,
            use_dinov3_encoder=use_dinov3_encoder,
        )

        predictor_models = {}
        if inferred_mode == "independent_dual":
            side_predictor, _, _, _, _ = load_checkpoint(
                r_path=predictor_checkpoint[0],
                predictor=copy.deepcopy(predictor),
                opt=None,
                scaler=None,
            )
            wrist_predictor, _, _, _, _ = load_checkpoint(
                r_path=predictor_checkpoint[1],
                predictor=copy.deepcopy(predictor),
                opt=None,
                scaler=None,
            )
            side_predictor.to(self.device).eval()
            wrist_predictor.to(self.device).eval()
            predictor_models["side"] = side_predictor
            predictor_models["wrist"] = wrist_predictor
        else:
            predictor, _, _, _, _ = load_checkpoint(
                r_path=predictor_checkpoint,
                predictor=predictor,
                opt=None,
                scaler=None,
            )
            predictor.to(self.device).eval()
            predictor_models[inferred_mode] = predictor

        encoder.to(self.device).eval()
        tokens_per_frame = int((crop_size // patch_size) ** 2)

        self.rollout_horizon = cfgs_mpc_args.get("rollout_horizon", 2)
        self.world_model = WorldModel(
            encoder=encoder,
            predictor=predictor_models,
            inferred_mode=inferred_mode,
            use_dinov3_encoder=use_dinov3_encoder,
            tokens_per_frame=tokens_per_frame,
            mpc_args={
                "rollout": self.rollout_horizon,
                "samples": cfgs_mpc_args.get("samples", 25),
                "topk": cfgs_mpc_args.get("topk", 10),
                "cem_steps": cfgs_mpc_args.get("cem_steps", 1),
                "momentum_mean": cfgs_mpc_args.get("momentum_mean", 0.15),
                "momentum_mean_gripper": cfgs_mpc_args.get("momentum_mean_gripper", 0.15),
                "momentum_std": cfgs_mpc_args.get("momentum_std", 0.75),
                "momentum_std_gripper": cfgs_mpc_args.get("momentum_std_gripper", 0.15),
                "maxnorm": cfgs_mpc_args.get("maxnorm", 0.075),
                "maxrotnorm": cfgs_mpc_args.get("maxrotnorm", 0.314),
                "verbose": cfgs_mpc_args.get("verbose", True),
                "objective": cfgs_exp_args.get("objective", "l1"),
                "warm_starting": cfgs_exp_args.get("warm-starting", False),
            },
            normalize_reps=True,
            device=self.device,
            transform=transform,
            log_objective_loss=cfgs_log_args.get("log_objective_loss", False),
        )

    def reset(self):
        self._env_prev_action.clear()
        self._env_goal_rep.clear()
        self._env_goal_rep_wrist.clear()

    def set_goal_images(self, external_image, wrist_image, *, env_id: int = 0, instruction: str = "goal"):
        goal_rep = self.world_model.encode(external_image)
        goal_rep_wrist = self.world_model.encode(wrist_image)
        self._env_goal_rep[env_id] = goal_rep
        self._env_goal_rep_wrist[env_id] = goal_rep_wrist
        self.world_model.reset_logs(
            goal_rep=goal_rep,
            goal_rep_wrist=goal_rep_wrist,
            exp_name=f"{instruction}_env{env_id}",
        )

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        import torch

        curr_obs = self._extract_observation(obs, env_id=env_id)

        if env_id not in self._env_goal_rep or env_id not in self._env_goal_rep_wrist:
            raise RuntimeError(
                "VALP goal images have not been set for this environment. "
                "Provide task goal images from the eval side before calling infer()."
            )

        with torch.no_grad():
            pose = torch.from_numpy(curr_obs["ee_pose"]).unsqueeze(0).to(
                self.device, dtype=torch.float32, non_blocking=True
            )

            actions, mean = self.world_model.infer_next_action(
                pose=pose,
                obs=curr_obs["external_image"],
                obs_wrist=curr_obs["wrist_image"],
                prev_action=self._env_prev_action.get(env_id),
            )

            self._env_prev_action[env_id] = mean[1:] if mean.shape[0] > 1 else None
            action = actions[0].detach().cpu().numpy().astype(np.float32)

        viz = np.concatenate(
            [
                curr_obs["external_image"].cpu().numpy(),
                curr_obs["wrist_image"].cpu().numpy(),
            ],
            axis=1,
        )
        print(action)
        return {"action": action, "viz": viz}

    def _extract_observation(self, obs_dict: dict, *, env_id: int) -> dict:
        from scipy.spatial.transform import Rotation

        robot_state = obs_dict["proprio_obs"]
        external_image = obs_dict["image_obs"]["external_right_cam"][env_id].clone().detach().cpu()
        wrist_image = obs_dict["image_obs"]["wrist_cam"][env_id].clone().detach().cpu()
        ee_pos = robot_state["ee_pos"][env_id].clone().detach().cpu().numpy()
        ee_quat = robot_state["ee_quat"][env_id].clone().detach().cpu().numpy()
        ee_rpy = Rotation.from_quat(ee_quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)
        gripper_pos = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()
        ee_pose = np.concatenate([ee_pos, ee_rpy, gripper_pos], axis=0).astype(np.float32)

        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "ee_pose": ee_pose,
        }

_HEADER = struct.Struct("!Q")

def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("VALP policy socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(sock: socket.socket, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)


def _recv_message(sock: socket.socket) -> dict:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))

class VALPDroidEEServer:
    """Tiny blocking TCP server that owns one loaded VALP policy."""

    def __init__(self, policy: VALPDroidEEPolicy, 
                        host: str = "0.0.0.0", 
                        port: int = 8000) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(1)
            print(f"[VALP server] Listening on {self.host}:{self.port}")

            while True:
                conn, addr = server_sock.accept()
                print(f"[VALP server] Client connected: {addr}")
                with conn:
                    while True:
                        try:
                            request = _recv_message(conn)
                            response = self._dispatch(request)
                        except ConnectionError:
                            print(f"[VALP server] Client disconnected: {addr}")
                            break
                        except Exception:
                            response = {"error": traceback.format_exc()}
                        try:
                            _send_message(conn, response)
                        except OSError:
                            print(f"[VALP server] Client disconnected: {addr}")
                            break

    def _dispatch(self, request: dict) -> dict:
        method = request.get("method", "infer")

        if method == "set_goal_images":
            self.policy.set_goal_images(
                request["external_image"],
                request["wrist_image"],
                env_id=int(request.get("env_id", 0)),
                instruction=request.get("instruction", "goal"),
            )
            return {"ok": True}

        if method == "reset":
            self.policy.reset()
            return {"ok": True}

        if method == "infer":
            return self.policy.infer(
                request["obs"], 
                request.get("instruction", ""), 
                env_id=int(request.get("env_id", 0))
            )

