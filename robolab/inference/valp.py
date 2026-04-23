# my_policy/inference_client.py

import numpy as np
from robolab.inference.base_client import InferenceClient


class MyPolicyClient(InferenceClient):
    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000) -> None:
        # Connect to your model server
        ...

    def infer(self, obs: dict, instruction: str) -> dict:
        # For the default DROID registration, obs contains:
        #   obs["image_obs"]["external_cam"]    - (N, H, W, 3) torch tensor, uint8
        #   obs["image_obs"]["wrist_cam"]       - (N, H, W, 3) torch tensor, uint8
        #   obs["proprio_obs"]["arm_joint_pos"] - (N, 7) torch tensor, float32
        #   obs["proprio_obs"]["gripper_pos"]   - (N, 1) torch tensor, float32

        # Extract observations for this env (N = num_envs; index by env_id)
        image = obs["image_obs"]["external_cam"][0].cpu().numpy()
        joint_pos = obs["proprio_obs"]["arm_joint_pos"][0].cpu().numpy()

        # Call your model server and get back an action
        action = self._query_server(image, joint_pos, instruction)

        # Return dict with "action" (np.ndarray) and "viz" (np.ndarray for display)
        return {
            "action": action,  # shape (8,): 7 joint positions + 1 gripper {0, 1}
            "viz": image,      # any RGB image for the live visualization window
        }

    def reset(self):
        # Called between episodes. Clear any internal state (action buffers, etc.)
        ...




    def __init__(
        self,
        cfg_path: str,
        model_name: str = "",
        default_checkpoint_path: str = "",
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        import yaml

        self.cfg_path = cfg_path
        with open(self.cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def initialize(self):
        # torch import
        import torch

        # VJEPA imports
        from app.vjepa_droid.transforms import make_transforms
        from inference.utils.world_model_wrapper import WorldModel

        self.device = self.cfg.get("device", "cuda")

        # model config 
        cfgs_model = self.cfg.get("model")

        pretrained_encoder = cfgs_model.get("pretrained_encoder", None)
        predictors = cfgs_model.get("predictors", None)

        self.side_decoder_name = cfgs_model.get("side_decoder", None)
        self.wrist_decoder_name = cfgs_model.get("wrist_decoder", None)

        # data config
        cfgs_data = self.cfg.get("data")
        crop_size = cfgs_data.get("crop_size", 256)
        patch_size = cfgs_data.get("patch_size", 16)
        tubelet_size = cfgs_data.get("tubelet_size", 2)

        # data augs
        cfgs_data_aug = self.cfg.get("data_aug")
        use_aa = cfgs_data_aug.get("auto_augment", False)
        horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
        motion_shift = cfgs_data_aug.get("motion_shift", False)
        ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
        rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
        reprob = cfgs_data_aug.get("reprob", 0.0)

        # cfgs_mpc_args config
        cfgs_mpc_args = self.cfg.get("mpc_args")
        self.rollout_horizon = cfgs_mpc_args.get("rollout_horizon", 2)
        samples = cfgs_mpc_args.get("samples", 25)
        topk = cfgs_mpc_args.get("topk", 10)
        cem_steps = cfgs_mpc_args.get("cem_steps", 1)
        momentum_mean = cfgs_mpc_args.get("momentum_mean", 0.15)
        momentum_mean_gripper = cfgs_mpc_args.get("momentum_mean_gripper", 0.15)
        momentum_std = cfgs_mpc_args.get("momentum_std", 0.75)
        momentum_std_gripper = cfgs_mpc_args.get("momentum_std_gripper", 0.15)
        maxnorm = cfgs_mpc_args.get("maxnorm", 0.075)
        maxrotnorm = cfgs_mpc_args.get("maxrotnorm", 0.314) 
        verbose = cfgs_mpc_args.get("verbose", True)

        # log
        cfgs_log_args = self.cfg.get("log")
        log_recons = cfgs_log_args.get("log_recons", False)
        log_objective_loss = cfgs_log_args.get("log_objective_loss", False)

        # exp
        cfgs_exp_args = self.cfg.get("exp")
        objective = cfgs_exp_args.get("objective", "l1")
        warm_starting = cfgs_exp_args.get("warm-starting", False)

        # Initialize transform (random-resize-crop augmentations)
        transform = make_transforms(
            random_horizontal_flip=horizontal_flip,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=crop_size,
        )

        # load pretrained encoder model
        encoder = torch.hub.load(
            ".", # path to hubconf.py
            pretrained_encoder, 
            source="local", 
            pretrained=True,
        )
        encoder.to(self.device).eval()
        tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

        # -- single predictors
        predictor_models = {}
        for predictor in predictors:
            pred_model = torch.hub.load(
                ".", # path to hubconf.py
                predictor, 
                source="local",
                encoder_embed_dim=encoder.embed_dim,
                pretrained=True,
            ) 
            pred_model.to(self.device).eval()
            predictor_models[predictor] = pred_model

        # -- side decoder
        side_decoder = None
        if log_recons:
            side_decoder = torch.hub.load(
                ".", # path to hubconf.py
                self.side_decoder_name, 
                source="local", 
                pretrained=True)
            side_decoder.to(self.device).eval()

        # -- wrist decoder
        wrist_decoder = None
        if log_recons:
            wrist_decoder = torch.hub.load(
                ".", # path to hubconf.py
                self.wrist_decoder_name, 
                source="local", 
                pretrained=True)
            wrist_decoder.to(self.device).eval()


        # World model wrapper initialization
        self.world_model = WorldModel(
            encoder=encoder,
            predictor=predictor_models,
            tokens_per_frame=tokens_per_frame,
            mpc_args={
                "rollout": self.rollout_horizon,
                "samples": samples,
                "topk": topk,
                "cem_steps": cem_steps,
                "momentum_mean": momentum_mean,
                "momentum_mean_gripper": momentum_mean_gripper,
                "momentum_std": momentum_std,
                "momentum_std_gripper": momentum_std_gripper,
                "maxnorm": maxnorm,
                "maxrotnorm": maxrotnorm,
                "verbose": verbose,
                "objective": objective,
                "warm_starting": warm_starting
            },
            normalize_reps=True,
            device=self.device,
            side_decoder = side_decoder,
            wrist_decoder = wrist_decoder,
            transform=transform,
            log_recons=log_recons,
            log_objective_loss=log_objective_loss,
        )

    def act(self, obs: Obs) -> Act:
        # torch imports
        import torch
        from torchvision.io import decode_jpeg

        with torch.no_grad():

            # [1, 7] -> [B, state_dim]
            # in DROID 0 is open to 1 is closed: float
            # In RCS 1 is open and 0 is close: binary
            print("received state", obs.info["xyzrpy"], 1-obs.gripper[0])
            s_n = (
                torch.tensor((np.concatenate(([obs.info["xyzrpy"], 
                                               [1-obs.gripper[0]]]), 
                                            axis=0))) 
                                            .unsqueeze(0)
                                            .to(self.device, 
                                            dtype=torch.float, 
                                            non_blocking=True)
                )

            side = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
            side = torch.frombuffer(bytearray(side), dtype=torch.uint8)
            side_img = decode_jpeg(side)

            wrist = base64.urlsafe_b64decode(obs.cameras["rgb_wrist"])
            wrist = torch.frombuffer(bytearray(wrist), dtype=torch.uint8)
            wrist_img = decode_jpeg(wrist)
  

            # Action conditioned predictor and zero-shot action inference with CEM
            actions, mean = self.world_model.infer_next_action(
                                            s_n,
                                            side_img,
                                            wrist_img, 
                                            prev_action=self.prev_action,
                                        ) # [rollout_horizon, 7]

            self.prev_action = mean[1:]
            
            first_action = actions[0].cpu()
            print(f"vjepa side action: {first_action.numpy()}")
            # convert back to RCS gripper format
            first_action[-1] =  1 - first_action[-1] 

        return Act(action=np.array(first_action))

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        super().reset(obs, instruction, **kwargs)
        # imports
        import torch
        from torchvision.io import decode_jpeg

        self.goal_rep = None

        goal_image = base64.urlsafe_b64decode(obs.cameras["rgb_side"])
        goal_image = torch.frombuffer(bytearray(goal_image), dtype=torch.uint8)
        self.goal_rep = self.world_model.encode(decode_jpeg(goal_image))
    
        goal_wrist = base64.urlsafe_b64decode(obs.cameras["rgb_wrist"])
        goal_wrist = torch.frombuffer(bytearray(goal_wrist), dtype=torch.uint8)
        self.goal_rep_wrist = self.world_model.encode(decode_jpeg(goal_wrist))

        self.prev_action = None
        if hasattr(self, "world_model"):
            self.world_model.reset_logs(goal_rep=self.goal_rep, 
                                        goal_rep_wrist=self.goal_rep_wrist,
                                        exp_name=instruction)

        return {}