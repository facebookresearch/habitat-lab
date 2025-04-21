import os
import sys

# Modify the path here for your robot-skills folder
VLA_PATH = (
    "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/repos/robot-skills"
)
sys.path.append(VLA_PATH)

try:
    # force the model to point to the new robot-skills
    sys.path.remove("__editable__.robot_skills-0.1.0.finder.__path_hook__")
except:
    pass
# Modify the path here for your transformers cache file
os.environ["TRANSFORMERS_CACHE"] = "/fsx-siro/jtruong/data/weights"
import json

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from src.model.vla.pizero import PiZeroInference
from src.model.vla.processing import VLAProcessor
from transformers import AutoTokenizer

from scripts.utils import hydra_solver_num_images_given_load_depth

# Only register once
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "num_images_given_load_depth", hydra_solver_num_images_given_load_depth
)
DEVICE = "cuda:0"

#### Result metric meta data ####
# in meter and degree threshold
SUCCESS_METRIC = [
    (1.0, 15),
    (1.0, 30),
    (1.0, 45),
    (1.0, 180),
    (2.0, 15),
    (2.0, 30),
    (2.0, 45),
    (2.0, 180),
    (3.0, 15),
    (3.0, 30),
    (3.0, 45),
    (3.0, 180),
    (4.0, 15),
    (4.0, 30),
    (4.0, 45),
    (4.0, 180),
    (5.0, 15),
    (5.0, 30),
    (5.0, 45),
    (5.0, 180),
]

#### Meta data for nav ####
# Data for clip the action and sensor
CLIP_MIN = -1
CLIP_MAX = 1

# VLA meta data
NUM_TOKEN_PER_IMAGE = 256
NUM_TEXT_TOKEN = 20


def plot_images(imgs, text):
    # imgs: [batch, channel, H, W]
    img = einops.rearrange(imgs[0], "C H W -> H W C") * 255
    img = img.to(torch.uint8)
    img = Image.fromarray(img.numpy())
    img.save(f"img_time_idx_{text}.png")


def diff(a1, a2):
    res = (a1 - a2) % 360
    if res < 180:
        return abs(res)
    else:
        return abs((360 - res))


def load_vla_skill_state_dict(path, model):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # Debug to check the parameter match
    # data["model"]["joint_model.mixtures.action.layers.14.self_attn.k_proj.weight"]
    # model.joint_model.mixtures.action.layers[14].self_attn.k_proj.weight
    # remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {
        k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
    }
    model.load_state_dict(data["model"], strict=True)
    # <All keys matched successfully>
    return model


def overwrite_config(config, main_config):
    """Manually overwrite the config variable here given the loaded model"""
    if "cond_steps" in main_config:
        config.cond_steps = main_config["cond_steps"]
    else:
        config.cond_steps = 10
    if "horizon_steps" in main_config:
        config.horizon_steps = main_config["horizon_steps"]
    else:
        config.horizon_steps = 5
    if "num_images_to_attend_in_the_past" in main_config:
        config.num_images_to_attend_in_the_past = main_config[
            "num_images_to_attend_in_the_past"
        ]
    else:
        config.num_images_to_attend_in_the_past = 0
    if "train_with_random_offset_position_ids" in main_config:
        config.train_with_random_offset_position_ids = main_config[
            "train_with_random_offset_position_ids"
        ]
    else:
        config.train_with_random_offset_position_ids = False
    if "action_history" in main_config:
        config.action_history = main_config["action_history"]
    else:
        config.action_history = False
    if "max_proprio_dim" in main_config:
        config.data.train.max_proprio_dim = main_config["max_proprio_dim"]
    else:
        config.data.train.max_proprio_dim = False
    if "use_flex" in main_config:
        config.use_flex = main_config["use_flex"]
    else:
        config.use_flex = True

    if "put_goal_in_action_model" in main_config:
        config.data.train.put_goal_in_action_model = main_config[
            "put_goal_in_action_model"
        ]
    else:
        config.data.train.put_goal_in_action_model = True

    if "num_images_per_step" in main_config:
        config.num_images_per_step = main_config["num_images_per_step"]
    else:
        config.num_images_per_step = 2

    if "only_use_depth" in main_config:
        config.data.train.only_use_depth = main_config["only_use_depth"]
    else:
        config.data.train.only_use_depth = False

    if "interleave_rgb_depth" in main_config:
        config.data.train.interleave_rgb_depth = main_config[
            "interleave_rgb_depth"
        ]
    else:
        config.data.train.interleave_rgb_depth = True

    if "is_point_nav_training" in main_config:
        config.data.train.is_point_nav_training = main_config[
            "is_point_nav_training"
        ]
    else:
        config.data.train.is_point_nav_training = True

    if "load_depth" in main_config:
        config.data.train.load_depth = main_config["load_depth"]
    else:
        config.data.train.load_depth = True

    if "ard_mode" in main_config:
        config.ard_mode = main_config["ard_mode"]
    else:
        config.ard_mode = False

    config.use_bf16 = True
    config.use_lm_head = True
    config.train_vlm = True
    config.data.train.skip_norm = False
    config.flow_sampling = "beta"
    return config


def get_statistics(config):
    # get dataset statistics:
    dataset_dir = os.path.join(
        config.data.train.data_path, config.data.train.dataset_mix, "1.0.0"
    )
    for file in os.listdir(dataset_dir):
        if file.startswith("dataset_statistics_") and file.endswith(".json"):
            statistics_path = os.path.join(dataset_dir, file)
            break
    with open(statistics_path, "r") as f:
        statistics = json.load(f)
    # Get the proprio and action statistics
    if "proprio" in statistics:
        PROPRIO_P01 = torch.tensor(statistics["proprio"]["p01"])
        PROPRIO_P99 = torch.tensor(statistics["proprio"]["p99"])
    if "action" in statistics:
        ACTION_P01 = np.array(statistics["action"]["p01"])
        ACTION_P99 = np.array(statistics["action"]["p99"])
    return PROPRIO_P01, PROPRIO_P99, ACTION_P01, ACTION_P99


def load_vla_skill(vla_skill_path, exp_name, main_config):
    """Load ckpt skill"""

    # Define the config path
    config = OmegaConf.load(
        os.path.join(VLA_PATH, "results", exp_name, "vla_cfg.yaml")
    )

    # Overwrite the meta config data for this vla_skill_path's model
    config = overwrite_config(config, main_config)
    global PROPRIO_P01, PROPRIO_P99, ACTION_P01, ACTION_P99
    PROPRIO_P01, PROPRIO_P99, ACTION_P01, ACTION_P99 = get_statistics(config)

    # Determine the model type
    dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float32

    # Initial the model and check if we want to distribute the model
    # to gpus
    if (
        "distribute_model_in_gpus" in main_config
        and main_config["distribute_model_in_gpus"]
    ):
        gpu_ids = []
        for i in range(torch.cuda.device_count()):
            gpu_ids.append(torch.device(f"cuda:{i}"))
        model = PiZeroInference(config, use_ddp=False, gpu_ids=gpu_ids)
    else:
        model = PiZeroInference(config, use_ddp=False)

    print(config)
    # Load the actual checkpoint
    model = load_vla_skill_state_dict(
        vla_skill_path,
        model,
    )

    # Freeze the weights
    model.freeze_all_weights()

    if "distribute_model_in_gpus" in main_config:
        if main_config["distribute_model_in_gpus"]:
            pass
        else:
            model.to(DEVICE)
    else:
        model.to(DEVICE)

    # To dtype
    model.to(dtype)

    # Compile the model
    model = torch.compile(
        model,
        mode="default",
    )
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )

    # Processor
    processor = VLAProcessor(
        tokenizer,
        config.vision.config.num_image_tokens,
        config.max_seq_len,
        config.tokenizer_padding,
    )

    return model, processor, config


def process_imgs(imgs, is_rgb=True):
    target_size = 224
    if is_rgb:
        # Resize the image here
        rgbs_process = torch.zeros(
            (imgs.shape[0], target_size, target_size, 3)
        )
        for i, rgb in enumerate(imgs):
            img = Image.fromarray(rgb.cpu().detach().numpy())
            img = img.resize((target_size, target_size))
            rgb = torch.as_tensor(np.array(img))
            rgbs_process[i] = rgb
        return rgbs_process
    else:
        imgs = einops.repeat(imgs, "B H W 1 -> B H W 3")
        depthes_process = torch.zeros(
            (imgs.shape[0], target_size, target_size, 3)
        )
        for i, depth in enumerate(imgs):
            img = Image.fromarray(depth.cpu().detach().numpy())
            img = img.resize((target_size, target_size))
            depth = torch.as_tensor(np.array(img))
            depthes_process[i] = depth
        return depthes_process


def normalize(data, p01, p99):
    """normalize the sensor: joint and action"""
    normalize_value = 2 * (data - p01) / (p99 - p01 + 1e-8) - 1
    return torch.clip(normalize_value, CLIP_MIN, CLIP_MAX)


def dennormalize(data, p01, p99):
    """denormalize the sensor: joint and action"""
    return (data - CLIP_MIN) / (CLIP_MAX - CLIP_MIN) * (p99 - p01) + p01


def alternate_input_len(vla_skill, vla_processor, dtype, new_cond_steps=20):
    """If we want to change the size of the input"""
    ######################## Experimental ########################
    ################ motify the input size on the fly ############
    # Rebuild the mask for the function that generates the mask
    vla_skill.max_image_text_tokens = (
        NUM_TOKEN_PER_IMAGE * new_cond_steps + NUM_TEXT_TOKEN
    )
    vla_skill.num_proprio_tokens = new_cond_steps
    vla_skill.total_num_tokens = (
        vla_skill.max_image_text_tokens
        + vla_skill.num_proprio_tokens
        + vla_skill.num_action_tokens
    )

    # Reinit vla processor
    vla_processor = VLAProcessor(
        vla_processor.tokenizer,
        NUM_TOKEN_PER_IMAGE * new_cond_steps,
        vla_skill.max_image_text_tokens,
        vla_processor.tokenizer_padding,
    )

    # Reinit siglip position stuff for positional encoder. This part needs to releart by
    # setting the max size like 100000
    vla_skill.vision_tower.vision_model.embeddings.position_ids = (
        torch.arange(NUM_TOKEN_PER_IMAGE * new_cond_steps)
        .expand((1, -1))
        .to(DEVICE)
    )
    vla_skill.vision_tower.vision_model.embeddings.position_embedding = (
        torch.nn.Embedding(
            NUM_TOKEN_PER_IMAGE * new_cond_steps,
            vla_skill.vision_tower.vision_model.embeddings.embed_dim,
        )
        .to(DEVICE)
        .to(dtype)
    )

    return vla_skill, vla_processor


def test_new_input_len(
    vla_skill, vla_processor, dtype, text, images, proprios, point_goals
):
    """Test the new input length"""

    images = torch.concatenate((images, images), dim=1)
    proprios = torch.concatenate((proprios, proprios), dim=1)
    cond_steps = 20
    images = einops.rearrange(
        images, "B T H W C -> (B T) C H W"
    )  # remove cond_steps dimension
    model_inputs = vla_processor(text=[text], images=images)
    model_inputs["pixel_values"] = einops.rearrange(
        model_inputs["pixel_values"],
        "(B T) C H W -> B T C H W",
        B=1,
        T=cond_steps,
    )

    (
        causal_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
    ) = vla_skill.build_causal_mask_and_position_ids(
        model_inputs["attention_mask"], dtype
    )

    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(dtype),
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": proprio_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": proprios.to(dtype),
        "point_goals": point_goals.to(dtype),
    }

    # For evaluation mode
    split_mask = True
    if split_mask:
        image_text_proprio_mask, action_mask = (
            vla_skill.split_full_mask_into_submasks(causal_mask)
        )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask
    else:
        inputs["causal_mask"] = causal_mask

    # Move the input to the main device
    inputs = {
        k: v.to(DEVICE) if type(v) != list else v for k, v in inputs.items()
    }

    # Finally, get the action
    # [batch_size, horizen, dim]
    preds = vla_skill.infer_action(**inputs)
    return preds


def infer_action(
    vla_skill,
    vla_processor,
    vla_config,
    observations,
    text,
    point_goal,
    get_next_num_actions=1,
    only_depth=False,
    interleave_rgb_depth=False,
    put_goal_in_action_model=True,
    is_point_nav_training=True,
):
    """Get the action given the input to VLA"""

    if vla_config.train_with_random_offset_position_ids:
        original_observation_length = len(observations)
        observations = observations[-50::]

    images = []
    # Get RGB images
    num_images_per_step = 1
    # observations is a list of dict
    for obs in observations:
        # torch.Size([1, 256, 256, 3]) -> torch.Size([1, 224, 224, 3])
        _image = process_imgs(
            obs["agent_0_articulated_agent_arm_rgb"], is_rgb=True
        )
        _image = _image.to(torch.uint8)
        images.append(_image)
    # torch.Size([1, 224, 224, 3]) -> torch.Size([1, cond_step, 224, 224, 3])
    images = torch.stack(images, dim=1)

    if vla_config.data.train.load_depth:
        num_images_per_step += 1
        # Get depth images
        depthes = []
        for obs in observations:
            # torch.Size([1, 224, 171, 1]) -> torch.Size([1, 224, 224, 3])
            _depth = obs["agent_0_articulated_agent_arm_depth"]
            if torch.max(_depth) <= 1:
                _depth = (255 * _depth).to(torch.uint8)
            else:
                _depth = _depth.to(torch.uint8)
            _depth = process_imgs(_depth, is_rgb=False)
            depthes.append(_depth.to(torch.uint8))

        # torch.Size([1, 224, 224, 3]) -> torch.Size([1, cond_step, 224, 224, 3])
        depthes = torch.stack(depthes, dim=1)
        # torch.Size([1, cond_step*2, 224, 224, 3])
        if interleave_rgb_depth:
            rgb_depth = []
            for i in range(depthes.shape[1]):
                rgb_depth.append(images[:, i])
                rgb_depth.append(depthes[:, i])
            # [batch, cond_step, H, W, C]
            images = torch.stack(rgb_depth, dim=1)
        elif only_depth:
            images = depthes
        else:
            images = torch.concatenate([images, depthes], axis=1)

    # Get the GPS
    proprios = []
    actions = []
    for i, obs in enumerate(observations):
        proprio = obs["agent_0_initial_gps_compass_sensor"]
        # Need to append the dimension
        if vla_config.proprio_dim != vla_config.data.train.max_proprio_dim:
            proprio = torch.concatenate(
                (
                    proprio,
                    torch.zeros(
                        1,
                        vla_config.data.train.max_proprio_dim
                        - vla_config.proprio_dim,
                    ),
                ),
                dim=1,
            )
        proprios.append(proprio)
        # Process action
        if i == len(observations) - 1:
            # In the final, we just pad zeros
            actions.append(torch.zeros(3))
        else:
            # In the curret i, we get the next action
            actions.append(observations[i + 1]["action"])

    # [batch_size, cond_step, proprio_dim]
    proprios = torch.stack(proprios, dim=1)
    # [cond_step, action_dim]
    actions = torch.stack(actions, dim=0)
    # [batch_size, cond_step, action_dim]
    actions = actions.unsqueeze(0)

    # TODO: check this
    # If we consider the history
    if vla_config.action_history:
        if not vla_config.data.train.skip_norm:
            actions[:, :, 0:2] = normalize(actions, ACTION_P01, ACTION_P99)[
                :, :, 0:2
            ]
        # The last element needs to be padded with zeros to indicate of prediction
        proprios[
            :, :-1, vla_config.proprio_dim : vla_config.proprio_dim + 3
        ] = actions[:, :-1]

    # Get the point nav goal
    # [batch_size, 3]
    point_goals = torch.tensor([point_goal])

    # Normalize the input to the skills
    if not vla_config.data.train.skip_norm:
        if vla_config.proprio_dim != vla_config.data.train.max_proprio_dim:
            # Only normalize the proprio
            proprios[:, :, 0 : vla_config.proprio_dim] = normalize(
                proprios[:, :, 0 : vla_config.proprio_dim],
                PROPRIO_P01,
                PROPRIO_P99,
            )
        else:
            proprios = normalize(proprios, PROPRIO_P01, PROPRIO_P99)
        # Goal is being normalized as well
        point_goals = normalize(point_goals, PROPRIO_P01, PROPRIO_P99)
    else:
        scale = 0.1
        point_goals = point_goals * scale
        proprios = proprios * scale

    # Get the condition step
    batch_size = images.shape[0]
    cond_steps = images.shape[1]

    if vla_config.data.train.load_depth and not only_depth:
        actual_cond_steps = int(cond_steps / 2)
    else:
        actual_cond_steps = cond_steps

    # build causal mask and position ids for action
    dtype = (
        torch.bfloat16 if vla_config.get("use_bf16", True) else torch.float32
    )

    # Determine if we want to keep the input length the same
    if vla_config.train_with_random_offset_position_ids:
        # Can vary the input length
        vla_skill, vla_processor = alternate_input_len(
            vla_skill, vla_processor, dtype, len(observations)
        )
    else:
        # Fixed input length
        if actual_cond_steps < vla_config.cond_steps:
            # need to repeat the observation so that it can get fit the causal mask
            if vla_config.data.train.load_depth and not only_depth:
                num_repeat = int(vla_config.cond_steps - int(cond_steps / 2))
            else:
                num_repeat = int(vla_config.cond_steps - cond_steps)
            if vla_config.data.train.load_depth and not only_depth:
                half = int(cond_steps / 2)
                rgb = images[:, 0:half]
                depth = images[:, half:]
                first_rgb = rgb[:, [0]]
                repeat_rgb = einops.repeat(
                    first_rgb, "B 1 H W C -> B R H W C", R=num_repeat
                )
                first_depth = depth[:, [0]]
                repeat_depth = einops.repeat(
                    first_depth, "B 1 H W C -> B R H W C", R=num_repeat
                )
                images = torch.concatenate(
                    (repeat_rgb, rgb, repeat_depth, depth), dim=1
                )
            else:
                # Get the first image
                first_image = images[:, [0]]
                repeat_image = einops.repeat(
                    first_image, "B 1 H W C -> B R H W C", R=num_repeat
                )
                images = torch.concatenate((repeat_image, images), dim=1)

            # Update the cond steps
            cond_steps = vla_config.cond_steps

            # for joint sensor
            first_proprio = proprios[:, [0]]
            repeat_proprio = einops.repeat(
                first_proprio, "B 1 D -> B R D", R=num_repeat
            )
            proprios = torch.concatenate((repeat_proprio, proprios), dim=1)

        elif actual_cond_steps > vla_config.cond_steps:
            if vla_config.data.train.load_depth and not only_depth:
                half = int(images.shape[1] / 2)
                rgb = images[:, 0:half]
                depth = images[:, half:]
                rgb = rgb[:, -vla_config.cond_steps : :]
                depth = depth[:, -vla_config.cond_steps : :]
                images = torch.concatenate((rgb, depth), dim=1)
            else:
                # Drop the first few frames
                images = images[:, -vla_config.cond_steps : :]
                # for joint sensor
            proprios = proprios[:, -vla_config.cond_steps : :]
            # Update the condition step
            cond_steps = vla_config.cond_steps

    ######### Experimental to overwrite the input length ########
    # test_new_input_len(vla_skill, vla_processor, dtype, text, images, proprios, point_goals)
    ######### Experimental to overwrite the input length ########

    images = einops.rearrange(
        images, "B T H W C -> (B T) C H W"
    )  # remove cond_steps dimension
    model_inputs = vla_processor(text=[text], images=images)

    print(f"point goal: {point_goals}")
    print(f"proprios: {proprios}")

    # Concate the point goal to proprios
    if not put_goal_in_action_model and is_point_nav_training:
        point_goals = einops.repeat(
            point_goals, "B D -> B C D", C=proprios.shape[1]
        )
        proprios[:, :, -3::] = point_goals

    if vla_config.train_with_random_offset_position_ids:
        model_inputs["pixel_values"] = einops.rearrange(
            model_inputs["pixel_values"],
            "(B T) C H W -> B T C H W",
            B=batch_size,
            T=(
                len(observations)
                if not (vla_config.data.train.load_depth and not only_depth)
                else int(len(observations) * 2)
            ),
        )
    else:
        model_inputs["pixel_values"] = einops.rearrange(
            model_inputs["pixel_values"],
            "(B T) C H W -> B T C H W",
            B=batch_size,
            T=(
                vla_config.cond_steps
                if not (vla_config.data.train.load_depth and not only_depth)
                else int(vla_config.cond_steps * 2)
            ),
        )

    # Debug
    # plot_images(images / 255, "test")

    (
        causal_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
    ) = vla_skill.build_causal_mask_and_position_ids(
        model_inputs["attention_mask"], dtype
    )

    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(dtype),
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": proprio_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": proprios.to(dtype),
        # "actions": actions.to(dtype),
    }
    if is_point_nav_training:
        inputs["point_goals"] = point_goals.to(dtype)  # type: ignore

    if vla_config.train_with_random_offset_position_ids:
        if original_observation_length > 50:
            offset = original_observation_length - 50
            inputs["vlm_position_ids"] = (
                inputs["vlm_position_ids"] + offset * 256
            )
            inputs["proprio_position_ids"] = (
                inputs["proprio_position_ids"] + offset
            )
            inputs["vision_tower_image_patch_position_ids"] = (
                vlm_position_ids[:, 0 : 256 * 50] - 1
            )
            print("proprio_position_ids", inputs["proprio_position_ids"])
            print(
                "proprio_position_ids.shape",
                inputs["proprio_position_ids"].shape,
            )
        else:
            inputs["vision_tower_image_patch_position_ids"] = (
                vlm_position_ids[:, 0 : 256 * len(observations)] - 1
            )

    # For evaluation mode
    split_mask = True
    if split_mask:
        image_text_proprio_mask, action_mask = (
            vla_skill.split_full_mask_into_submasks(causal_mask)
        )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask
    else:
        inputs["causal_mask"] = causal_mask

    # Move the input to the main device
    inputs = {
        k: v.to(DEVICE) if type(v) != list else v for k, v in inputs.items()
    }

    # Finally, get the action
    # [batch_size, horizen, dim]
    with torch.inference_mode():
        preds = vla_skill.infer_action(**inputs)

    if get_next_num_actions == 1:
        preds = preds[0, 0].cpu().detach().to(torch.float32)
    else:
        preds = (
            preds[0, 0:get_next_num_actions].cpu().detach().to(torch.float32)
        )

    print(f"raw vla pred: {preds}")

    if not vla_config.data.train.skip_norm:
        # Not need to do normalization on the done flag
        if get_next_num_actions == 1:
            preds[0:2] = dennormalize(preds, ACTION_P01, ACTION_P99)[0:2]
        else:
            preds[:, 0:2] = dennormalize(preds, ACTION_P01, ACTION_P99)[:, 0:2]
    else:
        # And the scale it back
        preds = preds / scale

    # Only depoly the first action, to float 32 conversion and numpy
    # [dim]
    preds = preds.numpy()

    print(f"normalized pred: {preds}")

    return preds


def performance_metric(infos):
    """Performance metric, check the distace for now"""
    dis = [v[-1] for v in infos]
    result = {}
    for m in SUCCESS_METRIC:
        if min(dis) <= m[0]:
            result[f"threshold:{m}"] = 1
        else:
            result[f"threshold:{m}"] = 0
    return result


def save_cur_performance_metric(infos, num_point, path):
    """Dump the result"""
    final = {}
    for key in infos:
        final[key] = infos[key] / num_point
    with open(f"{path}/result_log.json", "w") as f:
        json.dump(final, f)


def infer_action_pick(
    vla_skill,
    vla_processor,
    vla_config,
    obs,
    scale_input=True,
    descale_output=True,
    image_i=0,
    ard_mode=False,
):
    # RGB
    images = []
    _image = process_imgs(obs["image_raw"], is_rgb=True)
    _image = _image.to(torch.uint8)
    images.append(_image)
    # torch.Size([1, 224, 224, 3]) -> torch.Size([1, cond_step, 224, 224, 3])
    images = torch.stack(images, dim=1)
    if vla_config.data.train.load_depth:
        num_images_per_step += 1
        # Get depth images
        depthes = []
        # torch.Size([1, 224, 171, 1]) -> torch.Size([1, 224, 224, 3])
        _depth = obs["image_depth"]
        if torch.max(_depth) <= 1:
            _depth = (255 * _depth).to(torch.uint8)
        else:
            _depth = _depth.to(torch.uint8)
        _depth = process_imgs(_depth, is_rgb=False)
        depthes.append(_depth.to(torch.uint8))

        # torch.Size([1, 224, 224, 3]) -> torch.Size([1, cond_step, 224, 224, 3])
        depthes = torch.stack(depthes, dim=1)
        # torch.Size([1, cond_step*2, 224, 224, 3])
        # images = torch.concatenate([images, depthes], axis=1)

    # Joint
    proprios = []
    proprio = obs["proprio"]
    proprios.append(proprio)
    # [batch_size, cond_step, proprio_dim]
    proprios = torch.stack(proprios, dim=1)
    # Normalize
    if scale_input:
        proprios = normalize(proprios, PROPRIO_P01, PROPRIO_P99)

    # Get type
    dtype = (
        torch.bfloat16 if vla_config.get("use_bf16", True) else torch.float32
    )

    images = einops.rearrange(
        images, "B T H W C -> (B T) C H W"
    )  # remove cond_steps dimension

    # plot_images(images / 255, f"test_{image_i}")

    model_inputs = vla_processor(
        text=[obs["text"]] * batch_size, images=images
    )

    model_inputs["pixel_values"] = einops.rearrange(
        model_inputs["pixel_values"],
        "(B T) C H W -> B T C H W",
        B=batch_size,
        T=vla_config.cond_steps,
    )

    (
        causal_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
    ) = vla_skill.build_causal_mask_and_position_ids(
        model_inputs["attention_mask"], dtype
    )

    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(dtype),
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": proprio_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": proprios.to(dtype),
    }

    # For evaluation mode
    split_mask = True
    if split_mask:
        image_text_proprio_mask, action_mask = (
            vla_skill.split_full_mask_into_submasks(causal_mask)
        )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask
    else:
        inputs["causal_mask"] = causal_mask

    # Move the input to the main device
    inputs = {
        k: v.to(DEVICE) if type(v) != list else v for k, v in inputs.items()
    }
    # Finally, get the action
    # [batch_size, horizen, dim] = torch.Size([1, 1, 23])
    with torch.inference_mode():
        if ard_mode:
            preds = vla_skill.infer_action_ard_mode(**inputs)
        else:
            preds = vla_skill.infer_action(**inputs)

    # The action was dennormalized
    preds = preds.cpu().detach().to(torch.float32)
    if descale_output:
        preds = dennormalize(preds, ACTION_P01, ACTION_P99)
    return preds


def infer_action_open_murp(
    vla_skill,
    vla_processor,
    vla_config,
    obs,
    scale_input=True,
    descale_output=True,
    image_i=0,
    ard_mode=False,
):
    # RGB
    images = []
    _image = process_imgs(obs["image_raw"], is_rgb=True)
    _image = _image.to(torch.uint8)
    images.append(_image)
    # torch.Size([1, 224, 224, 3]) -> torch.Size([1, cond_step, 224, 224, 3])
    images = torch.stack(images, dim=1)
    if vla_config.data.train.load_depth:
        # num_images_per_step += 1
        # Get depth images
        depthes = []
        # torch.Size([1, 224, 171, 1]) -> torch.Size([1, 224, 224, 3])
        _depth = obs["image_depth"]
        if torch.max(_depth) <= 1:
            _depth = (255 * _depth).to(torch.uint8)
        else:
            _depth = _depth.to(torch.uint8)
        _depth = process_imgs(_depth, is_rgb=False)

        depthes.append(_depth.to(torch.uint8))

        # torch.Size([1, 224, 224, 3]) -> torch.Size([1, cond_step, 224, 224, 3])
        depthes = torch.stack(depthes, dim=1)
        # torch.Size([1, cond_step*2, 224, 224, 3])
        # images = torch.concatenate([images, depthes], axis=1)
        images = torch.concatenate((images, images, depthes, depthes), dim=1)
    else:
        images = torch.concatenate((images, images), dim=1)
    # images = torch.concatenate((images, images, depthes, depthes), dim=1)
    batch_size = images.shape[0]

    # Joint
    proprios = []
    proprio = obs["proprio"]
    proprios.append(proprio)
    # [batch_size, cond_step, proprio_dim]
    proprios = torch.stack(proprios, dim=1)
    proprios = torch.concatenate((proprios, proprios), dim=1)
    # Normalize
    if scale_input:
        proprios = normalize(proprios, PROPRIO_P01, PROPRIO_P99)

    # Get type
    dtype = (
        torch.bfloat16 if vla_config.get("use_bf16", True) else torch.float32
    )

    images = einops.rearrange(
        images, "B T H W C -> (B T) C H W"
    )  # remove cond_steps dimension

    # plot_images(images / 255, f"test_{image_i}")

    model_inputs = vla_processor(
        text=[obs["text"]] * batch_size, images=images
    )

    model_inputs["pixel_values"] = einops.rearrange(
        model_inputs["pixel_values"],
        "(B T) C H W -> B T C H W",
        B=batch_size,
        # T=vla_config.cond_steps * 2,
        T=(
            vla_config.cond_steps
            if not (vla_config.data.train.load_depth)
            else vla_config.cond_steps * 2
        ),
    )

    (
        causal_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
    ) = vla_skill.build_causal_mask_and_position_ids(
        model_inputs["attention_mask"], dtype
    )

    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(dtype),
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": proprio_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": proprios.to(dtype),
    }

    # For evaluation mode
    split_mask = True
    if split_mask:
        image_text_proprio_mask, action_mask = (
            vla_skill.split_full_mask_into_submasks(causal_mask)
        )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask
    else:
        inputs["causal_mask"] = causal_mask

    # Move the input to the main device
    inputs = {
        k: v.to(DEVICE) if type(v) != list else v for k, v in inputs.items()
    }
    # for k, v in inputs.items():
    # print(k, v.shape)
    # Finally, get the action
    # [batch_size, horizen, dim] = torch.Size([1, 1, 23])
    with torch.inference_mode():
        if ard_mode:
            preds = vla_skill.infer_action_ard_mode(**inputs)
        else:
            preds = vla_skill.infer_action(**inputs)

    # The action was dennormalized
    preds = preds.cpu().detach().to(torch.float32)
    if descale_output:
        preds = dennormalize(preds, ACTION_P01, ACTION_P99)
    return preds
