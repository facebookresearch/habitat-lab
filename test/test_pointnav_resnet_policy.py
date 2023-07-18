#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import shlex
import subprocess

import numpy as np
import pytest
import torch
from gym import spaces

from habitat import read_write
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

ACTION_SPACE = spaces.Discrete(4)

OBSERVATION_SPACES = {
    "depth_model": spaces.Dict(
        {
            "depth": spaces.Box(
                low=0,
                high=1,
                shape=(256, 256, 1),
                dtype=np.float32,
            ),
            "pointgoal_with_gps_compass": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    ),
    "rgb_model": spaces.Dict(
        {
            "rgb": spaces.Box(
                low=0,
                high=255,
                shape=(256, 256, 3),
                dtype=np.uint8,
            ),
            "pointgoal_with_gps_compass": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    ),
    "blind_model": spaces.Dict(
        {
            "pointgoal_with_gps_compass": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    ),
}

MODELS_DEST_DIR = "data/ddppo-models"
MODELS_BASE_URL = "https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models"
MODELS_TO_TEST = {
    "gibson-2plus-resnet50.pth": {
        "backbone": "resnet50",
        "observation_space": OBSERVATION_SPACES["depth_model"],
        "action_space": ACTION_SPACE,
    },
    "gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth": {
        "backbone": "se_resneXt50",
        "observation_space": OBSERVATION_SPACES["rgb_model"],
        "action_space": ACTION_SPACE,
    },
    "gibson-0plus-mp3d-train-val-test-blind.pth": {
        "backbone": None,
        "observation_space": OBSERVATION_SPACES["blind_model"],
        "action_space": ACTION_SPACE,
    },
}


def _get_model_url(model_name):
    return f"{MODELS_BASE_URL}/{model_name}"


def _get_model_path(model_name):
    return f"{MODELS_DEST_DIR}/{model_name}"


@pytest.fixture(scope="module", autouse=True)
def download_data():
    for model_name in MODELS_TO_TEST:
        model_url = _get_model_url(model_name)
        model_path = _get_model_path(model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}.")
            download_command = (
                "wget --continue " + model_url + " -P " + MODELS_DEST_DIR
            )
            subprocess.check_call(shlex.split(download_command))
            assert os.path.exists(
                model_path
            ), "Download failed, no package found."


@pytest.mark.parametrize(
    "pretrained_weights_path,backbone,observation_space,action_space",
    [
        (
            _get_model_path(model_name),
            params["backbone"],
            params["observation_space"],
            params["action_space"],
        )
        for model_name, params in MODELS_TO_TEST.items()
    ],
)
def test_pretrained_models(
    pretrained_weights_path, backbone, observation_space, action_space
):
    config = get_config(
        "test/config/habitat_baselines/ddppo_pointnav_test.yaml"
    )
    with read_write(config):
        ddppo_config = config.habitat_baselines.rl.ddppo
        ddppo_config.pretrained = True
        ddppo_config.pretrained_weights = pretrained_weights_path
        if backbone is not None:
            ddppo_config.backbone = backbone

    policy = PointNavResNetPolicy.from_config(
        config=config,
        observation_space=observation_space,
        action_space=action_space,
    )

    pretrained_state = torch.load(pretrained_weights_path, map_location="cpu")

    prefix = "actor_critic."
    policy.load_state_dict(
        {  # type: ignore
            k[len(prefix) :]: v
            for k, v in pretrained_state["state_dict"].items()
        }
    )
