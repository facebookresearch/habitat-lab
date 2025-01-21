#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example script to load robot skills models into habitat environment
"""

import math
import sys

PATH_TO_ROBOT_SKILLS = "/data/home/jimmytyyang/facebook/vla/robot-skills"
# Link your path to robot-skills repo
sys.path.append(PATH_TO_ROBOT_SKILLS)
from omegaconf import OmegaConf
from src.model.vla.processing import VLAProcessor
from transformers import AutoTokenizer

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

import numpy as np
import torch
from PIL import Image
from src.model.vla.pizero import PiZero

# Also make sure you do the following to unset the path and export
# needed path or add these into bashrc file.
# unset LD_LIBRARY_PATH
# export TRANSFORMERS_CACHE=<your_path>
# export VLA_DATA_DIR=<your_path>
# export VLA_LOG_DIR=<your_path>


def example():
    """The example here shows how to initialize the model and use that to describe the image given the text prompt"""
    # Define the config path
    config = OmegaConf.load(
        f"{PATH_TO_ROBOT_SKILLS}/config/train/mg97hv104eval_jan12v2_0.yaml"
    )
    # We use LLM mode here for demostration purpose
    config.cond_steps = 1
    config.use_lm_head = True
    config.mixture.vlm.use_final_norm = True
    # Load the model
    model = PiZero(config)
    model.tie_action_proprio_weights()
    model.load_pretrained_weights()
    dtype = torch.bfloat16
    device = "cuda"
    model.to("cuda")
    model.to(dtype)
    model.eval()

    # Dummy image, and it will be replaced by the first image with a real one
    bsz = 1
    dummy_images = torch.randint(
        0, 256, (bsz, 3, 224, 224), dtype=torch.uint8
    )  # not used if text_only
    real_image_path = f"{PATH_TO_ROBOT_SKILLS}/media/maniskill_pp.png"
    real_image = Image.open(real_image_path).convert("RGB")
    real_image_t = torch.as_tensor(
        np.array(real_image.resize((224, 224))).transpose(2, 0, 1)
    )
    dummy_images[0] = real_image_t

    # Text and proprio
    dummy_texts = [
        "this image shows ",
        "this is a nice portrait of London because ",
    ][:bsz]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    assert tokenizer.padding_side == "right"

    # Processor
    num_image_tokens = config.vision.config.num_image_tokens
    processor = VLAProcessor(tokenizer, num_image_tokens, config.max_seq_len)

    # Process image and text
    model_inputs = processor(text=dummy_texts, images=dummy_images)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"].to(dtype)

    kv_cache = model.build_text_cache()
    num_tokens_to_generate = 20
    print(f"Generating text of maximum {num_tokens_to_generate} tokens...")

    # Start to generate the text token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    for _ in range(num_tokens_to_generate):
        with torch.inference_mode():
            outputs = model.infer_text(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(device),
                attention_mask=attention_mask.to(device),
                kv_cache=kv_cache,
            )
        next_token_logits = outputs["logits"][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Only input the new token the next time since using cache
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)],
            dim=-1,
        )
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    print("=========================")
    print("Image path:", real_image_path)
    print("Prompt:", dummy_texts[0])
    print("Generated text:", decoded)


if __name__ == "__main__":
    example()
