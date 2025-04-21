import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

third_party_ckpt_root_folder = "/fsx-siro/jtruong/repos/robot-skills"
sys.path.append(third_party_ckpt_root_folder)


# robot-skills related import
from src.model.vla.pizero import PiZeroInference
from src.model.vla.processing import VLAProcessor
from transformers import AutoTokenizer


class VLAPolicy:
    """
    Loading policy
    """

    def __init__(self, config, device):
        np.set_printoptions(precision=4, suppress=True)
        torch.set_printoptions(precision=4, sci_mode=False)
        self.observation_dict: List[Any] = (
            []
        )  # This is to store the past observations from habitat
        self.vla_action: List[Any] = (
            []
        )  # This is to store the actions from action chunk
        self.depoly_one_action = True  # If we want to do MPC style -- only depoly one action from action chunk
        # TODO: expand these to multi-sensors
        self.vla_target_image = "arm_rgb"  # Target RGB
        self.vla_target_proprio = "joint"  # Target proprio sensor
        # vla_target_proprio = "ee_pos"  # Target proprio sensor
        # load checkpoint
        self.config = config
        self.device = device

        (
            self.vla_model,
            self.vla_processor,
            self.vla_config,
        ) = self.load_process_robot_skills_ckpt(
            config.VLA_CKPT,
            config.VLA_CONFIG_FILE,
        )

        self.reset()

    @staticmethod
    def process_rgb(rgbs, target_size):
        """Resize the rgb images"""
        # Resize the image here
        rgbs_process = torch.zeros(
            (rgbs.shape[0], 3, target_size, target_size)
        )
        for i, rgb in enumerate(rgbs):
            img = Image.fromarray(rgb)
            img = img.resize((target_size, target_size))
            img = np.array(img)
            rgb = torch.as_tensor(
                rearrange(img, "h w c-> c h w")
            )  # torch.Size([3, 224, 224])
            rgbs_process[i] = rgb
        return rgbs_process

    @staticmethod
    def load_robot_skills_ckpt(path, model):
        """load the torch checkpoint"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        # remove "_orig_mod." prefix if saved model was compiled
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }
        model.load_state_dict(data["model"], strict=True)
        return model

    # Load policy
    def load_process_robot_skills_ckpt(
        self,
        load_vla_ckpt,
        third_party_config_path_dir,
    ):
        """Load and process robot skills ckpt"""
        # Process the config
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver("round_up", math.ceil)
        OmegaConf.register_new_resolver("round_down", math.floor)

        # Define the config path
        config = OmegaConf.load(
            third_party_config_path_dir,
        )

        # Make sure your config here is correct. We can also do config overwrite here if
        # you want to do a quick hack
        config.cond_steps = 2
        config.use_lm_head = True
        config.mixture.vlm.use_final_norm = True
        config.horizon_steps = 4

        # Load the model
        model = PiZeroInference(config, use_ddp=False)
        model = self.load_robot_skills_ckpt(
            load_vla_ckpt,
            model,
        )

        # Housekeeping the model
        model.freeze_all_weights()
        model.to(self.device)
        model.to(torch.bfloat16)  # Save memeory
        model = torch.compile(
            model,
            mode="default",
        )
        model.eval()

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_path, padding_side="right"
        )

        # processor
        processor = VLAProcessor(
            tokenizer,
            config.vision.config.num_image_tokens,
            config.max_seq_len,
            config.tokenizer_padding,
        )

        return model, processor, config

    def reset(self):
        self.images = torch.randint(
            0,
            256,
            (
                1,
                self.vla_config.cond_steps,
                3,
                self.vla_config.image_size,
                self.vla_config.image_size,
            ),
            dtype=torch.uint8,
        )
        self.proprio = torch.zeros(
            (1, self.vla_config.cond_steps, self.vla_config.proprio_dim)
        )
        self.texts = [self.config.LANGUAGE_INSTRUCTION]
        self.observation_dict = []

    def infer_action_vla_model(
        self,
        vla_model,
        processor,
        observation,
        device,
        vla_config,
    ):
        """Infer action using vla models."""
        self.observation_dict.append(observation)
        # Confirm the number of batches
        if len(observation[self.vla_target_image].shape) > 3:
            observation[self.vla_target_image] = np.transpose(
                observation[self.vla_target_image], (3, 0, 1, 2)
            )
        else:
            observation[self.vla_target_image] = np.expand_dims(
                observation[self.vla_target_image], axis=0
            )
        observation[self.vla_target_proprio] = torch.as_tensor(
            observation[self.vla_target_proprio]
        )
        bsz = observation[self.vla_target_image].shape[0]

        self.images = torch.randint(
            0,
            256,
            (
                1,
                self.vla_config.cond_steps,
                3,
                self.vla_config.image_size,
                self.vla_config.image_size,
            ),
            dtype=torch.uint8,
        )
        self.proprio = torch.zeros(
            (1, self.vla_config.cond_steps, self.vla_config.proprio_dim)
        )

        if len(self.observation_dict) < vla_config.cond_steps:
            store_size = len(self.observation_dict)
            for i in range(store_size):
                self.images[:, vla_config.cond_steps - i - 1] = (
                    self.process_rgb(
                        self.observation_dict[-i - 1][self.vla_target_image],
                        vla_config.image_size,
                    )
                )
                if self.vla_target_proprio == "ee_pos":
                    prop_obs = self.observation_dict[-i - 1][
                        self.vla_target_proprio
                    ][:, :3]
                else:
                    prop_obs = self.observation_dict[-i - 1][
                        self.vla_target_proprio
                    ]
                self.proprio[:, vla_config.cond_steps - i - 1] = prop_obs
            # Pad the image one with the last image
            for i in range(vla_config.cond_steps - store_size):
                self.images[:, i] = self.process_rgb(
                    self.observation_dict[0][self.vla_target_image],
                    vla_config.image_size,
                )
                if self.vla_target_proprio == "ee_pos":
                    prop_obs = self.observation_dict[-i - 1][
                        self.vla_target_proprio
                    ][:, :3]
                else:
                    prop_obs = self.observation_dict[-i - 1][
                        self.vla_target_proprio
                    ]

                self.proprio[:, i] = prop_obs
        else:
            for i in range(vla_config.cond_steps):
                self.images[:, i] = self.process_rgb(
                    self.observation_dict[i - vla_config.cond_steps][
                        self.vla_target_image
                    ],
                    vla_config.image_size,
                )
                if self.vla_target_proprio == "ee_pos":
                    prop_obs = self.observation_dict[
                        i - vla_config.cond_steps
                    ]["ee_pose"][:, :3]
                else:
                    prop_obs = self.observation_dict[
                        i - vla_config.cond_steps
                    ][self.vla_target_proprio]

                self.proprio[:, i] = prop_obs

        self.images = rearrange(self.images, "B T C H W -> (B T) C H W")

        dtype = torch.bfloat16
        # process image and text
        model_inputs = processor(text=self.texts, images=self.images)
        model_inputs["pixel_values"] = rearrange(
            model_inputs["pixel_values"],
            "(B T) C H W -> B T C H W",
            B=bsz,
            T=vla_config.cond_steps,
        )
        (
            causal_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
        ) = vla_model.build_causal_mask_and_position_ids(
            model_inputs["attention_mask"], dtype=dtype
        )
        (
            image_text_proprio_mask,
            action_mask,
        ) = vla_model.split_full_mask_into_submasks(causal_mask)

        with torch.inference_mode():
            actions = vla_model.infer_action(
                input_ids=model_inputs["input_ids"].to(device),
                pixel_values=model_inputs["pixel_values"].to(dtype).to(device),
                image_text_proprio_mask=image_text_proprio_mask.to(device),
                action_mask=action_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=self.proprio.to(dtype).to(device),
            )  # [bsz, horizon, dim]
        return actions

    def act(self, batch):
        # get action
        self.vla_action = []
        vla_action = self.infer_action_vla_model(
            self.vla_model,
            self.vla_processor,
            batch,
            self.device,
            self.vla_config,
        )
        # Make the time horizon as a leading dimension
        vla_action = rearrange(vla_action, "b h a-> h b a")
        for vla_a_time in vla_action:
            vla_temp = []
            for vla_a_batch in vla_a_time:
                vla_temp.append(vla_a_batch.cpu().detach().float().numpy())
            self.vla_action.append(vla_temp)
        return self.vla_action
