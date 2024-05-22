#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import path as osp

import magnum
import pytest
from hydra import compose, initialize

from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins


class AppStateTest(AppState):
    """
    A minimal HITL test app that loads and steps a Habitat environment, with
    a fixed overhead camera.
    """

    def __init__(self, app_service):
        self._app_service = app_service

    def sim_update(self, dt, post_sim_update_dict):
        assert not self._app_service.env.episode_over
        self._app_service.compute_action_and_step_env()

        # set the camera for the main 3D viewport
        post_sim_update_dict["cam_transform"] = magnum.Matrix4.look_at(
            eye=magnum.Vector3(-20, 20, -20),
            target=magnum.Vector3(0, 0, 0),
            up=magnum.Vector3(0, 1, 0),
        )


def main(config) -> None:
    hitl_main(config, lambda app_service: AppStateTest(app_service))


@pytest.mark.skipif(
    not osp.exists("data/scene_datasets/hssd-hab"),
    reason="Requires public Habitat-HSSD scene dataset. TODO: should be updated to a new dataset.",
)
def test_hitl_main():
    register_hydra_plugins()
    with initialize(version_base=None, config_path="config"):
        cfg = compose(
            config_name="base_test_cfg",
            overrides=[
                "+experiment=smoke_test",
            ],
        )

    main(cfg)
