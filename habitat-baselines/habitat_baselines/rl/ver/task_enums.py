#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum


class EnvironmentWorkerTasks(enum.Enum):
    start = enum.auto()
    step = enum.auto()
    reset = enum.auto()
    set_transfer_buffers = enum.auto()
    set_action_plugin = enum.auto()
    start_experience_collection = enum.auto()
    wait = enum.auto()


class ReportWorkerTasks(enum.Enum):
    episode_end = enum.auto()
    learner_update = enum.auto()
    learner_timing = enum.auto()
    env_timing = enum.auto()
    policy_timing = enum.auto()
    start_collection = enum.auto()
    state_dict = enum.auto()
    load_state_dict = enum.auto()
    preemption_decider = enum.auto()
    num_steps_collected = enum.auto()
    get_window_episode_stats = enum.auto()


class PreemptionDeciderTasks(enum.Enum):
    policy_step = enum.auto()
    start_rollout = enum.auto()
    end_rollout = enum.auto()
    learner_time = enum.auto()


class InferenceWorkerTasks(enum.Enum):
    set_rollouts = enum.auto()
    set_actor_critic_tensors = enum.auto()
    start = enum.auto()
