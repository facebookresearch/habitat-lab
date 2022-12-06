# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from gym import spaces
from gym.vector.utils.spaces import batch_space

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (  # get_active_obs_transforms,
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
)
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    CenterCropperConfig,
    Cube2EqConfig,
    Cube2FishConfig,
    Eq2CubeConfig,
    ObsTransformConfig,
    ResizeShortestEdgeConfig,
)


@pytest.mark.parametrize(
    "obs_transform_config",
    [
        ResizeShortestEdgeConfig(),
        CenterCropperConfig(),
        Cube2EqConfig(),
        Cube2FishConfig(),
        Eq2CubeConfig(),
    ],
)
def test_transforms(obs_transform_config: ObsTransformConfig):
    transformer_cls = baseline_registry.get_obs_transformer(
        obs_transform_config.type
    )
    transformer = transformer_cls.from_config(obs_transform_config)
    obs_space = spaces.Dict(
        {
            "BACK": spaces.Box(low=0, high=1, shape=(32, 16, 3)),
            "DOWN": spaces.Box(low=0, high=1, shape=(32, 16, 3)),
            "FRONT": spaces.Box(low=0, high=1, shape=(32, 16, 3)),
            "LEFT": spaces.Box(low=0, high=1, shape=(32, 16, 3)),
            "RIGHT": spaces.Box(low=0, high=1, shape=(32, 16, 3)),
            "UP": spaces.Box(low=0, high=1, shape=(32, 16, 3)),
        }
    )
    modified_obs_space = transformer.transform_observation_space(
        observation_space=obs_space
    )
    assert modified_obs_space == apply_obs_transforms_obs_space(
        obs_space, [transformer]
    )
    batched_obs_space = batch_space(obs_space, 7)
    observation = batched_obs_space.sample()
    tensor_observation = TensorDict.from_tree(observation)
    transformed_obs = apply_obs_transforms_batch(
        tensor_observation, [transformer]
    )
    assert modified_obs_space.contains(
        {k: v[0] for k, v in transformed_obs.items()}
    ), f"Observation transform generated the observation ({str({k: v.shape for k,v in transformed_obs.items()}) }) which is incompatible with the defined observation space {modified_obs_space}"
