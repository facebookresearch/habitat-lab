#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from habitat.config.default import get_config
from habitat.core.benchmark import Benchmark
from habitat.tasks.rearrange.sub_tasks.pick_sensors import RearrangePickSuccess
from habitat_baselines.agents.mp_agents import (
    AgentComposition,
    SpaManipPick,
    SpaResetModule,
)
from habitat_baselines.motion_planning.motion_plan import is_ompl_installed

TEST_CFG = "habitat_baselines/config/rearrange/spap_pick.yaml"


@pytest.mark.skipif(
    not is_ompl_installed(),
    reason="The Open Motion Planning Library is not installed.",
)
def test_pick_motion_planning():
    config = get_config(TEST_CFG)

    benchmark = Benchmark(config.BASE_TASK_CONFIG_PATH)

    def get_args(skill):
        target_idx = skill._sim.get_targets()[0][0]
        return {"obj": target_idx}

    ac_cfg = get_config(config.BASE_TASK_CONFIG_PATH).TASK.ACTIONS
    spa_cfg = config.SENSE_PLAN_ACT
    env = benchmark._env
    pick_skill = AgentComposition(
        [
            SpaManipPick(env, spa_cfg, ac_cfg, auto_get_args_fn=get_args),
            SpaResetModule(
                env,
                spa_cfg,
                ac_cfg,
                ignore_first=True,
                auto_get_args_fn=get_args,
            ),
        ],
        env,
        spa_cfg,
        ac_cfg,
        auto_get_args_fn=get_args,
    )
    metrics = benchmark.evaluate(pick_skill, 1)
    assert metrics[RearrangePickSuccess.cls_uuid] == 1.0
