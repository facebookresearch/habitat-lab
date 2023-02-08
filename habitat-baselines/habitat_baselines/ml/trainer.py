#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
import tqdm

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer

# from habitat_baselines.common.obs_transformers import (
#     apply_obs_transforms_batch,
#     get_active_obs_transforms,
# )

# from habitat_baselines.utils.common import (
#     batch_obs,
#     get_num_actions,
#     is_continuous_action_space,
# )


@baseline_registry.register_trainer(name="ml")
class MLTrainer(PPOTrainer):
    r"""Trainer class for modular baseline
    Paper: https://arxiv.org/abs/2007.00643.
    """

    def __init__(self, config=None):
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        print(config)
        print("ML trainer initialized.")

    def eval(
        self,
    ) -> None:
        r"""Evaluates a single checkpoint.
        Args:
            checkpoint_path: path of checkpoint

        Returns:
            None
        """

        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # self.obs_transforms = get_active_obs_transforms(self.config)

        self._init_envs()
        action_space = self.envs.action_spaces[0]

        print("Starting eval.")

        # if is_continuous_action_space(action_space):
        #     # Assume NONE of the actions are discrete
        #     action_shape = (get_num_actions(action_space),)
        #     discrete_actions = False
        # else:
        #     # For discrete pointnav
        #     action_shape = (1,)
        #     discrete_actions = True

        # [TODO] manually overriding action space
        action_shape = 7
        discrete_actions = True

        observations = self.envs.reset()
        # batch = batch_obs(obs, device=self.device)

        policy = baseline_registry.get_policy(
            self.config.habitat_baselines.ml.policy.name
        )
        self.policy = policy.from_config(self.config, self.envs, self.device)

        # batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]  # type: List[List[np.ndarray]]

        number_of_eval_episodes = (
            self.config.habitat_baselines.test_episode_count
        )

        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        stats_episodes = []
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and
            self.envs.num_envs
            > 0
        ):
            current_episodes_info = self.envs.current_episodes()
            obs, infos = zip(
                *self.envs.call(
                    ["preprocess_obs"] * self.envs.num_envs,
                    [{"obs": observation} for observation in observations],
                )
            )
            actions = self.policy.act(obs, infos, self.envs)
            step_data = actions
            outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for e, done in enumerate(dones):
                if done:
                    self.policy.reset_vectorized_for_env(e)
                    print(infos[e])
                    stats_episodes.append(infos[e])