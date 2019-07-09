import argparse
import glob
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from habitat import Config, get_config, make_dataset
from habitat_baselines.config.default import get_config as baseline_cfg


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def _flatten_helper(t, n, tensor):
    return tensor.view(t * n, *tensor.size()[2:])


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


def _flatten_helper(t, n, tensor):
    return tensor.view(t * n, *tensor.size()[2:])


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    r"""Decreases the learning rate linearly
    """
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser


def get_exp_config(cfg_path, opts=None):
    config = Config(new_allowed=True)
    config.merge_from_other_cfg(baseline_cfg(cfg_path))
    print(config)
    config.merge_from_other_cfg(get_config(config.BASELINE.RL.PPO.task_config))
    if opts is not None:
        config.merge_from_list(opts)
    return config


def make_env_fn(config, env_class, rank):
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    config.defrost()
    config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config.freeze()
    env = env_class(
        config_env=config, config_baseline=config.BASELINE, dataset=dataset
    )
    env.seed(rank)
    return env


def batch_obs(observations):
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(obs[sensor])

    for sensor in batch:
        batch[sensor] = torch.tensor(
            np.array(batch[sensor]), dtype=torch.float
        )
    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), "invalid checkpoint folder path"
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None
