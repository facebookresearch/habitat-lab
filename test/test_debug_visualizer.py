#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import pytest

import habitat_sim
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_sim.utils.settings import default_sim_settings, make_cfg


@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_debug_visualizer():
    ######################
    # NOTE: set show_images==True to see the output images from the various dbv features
    show_images = False
    ######################

    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)
    with habitat_sim.Simulator(hab_cfg) as sim:
        # at first sim should have only the initial default rgb sensor and agent
        assert len(sim._Simulator__sensors) == 1
        assert len(sim.agents) == 1

        # initialize the dbv
        dbv = DebugVisualizer(sim)

        # before initializing nothing changes
        assert len(sim._Simulator__sensors) == 1
        assert len(sim.agents) == 1

        # create and register the agent/sensor
        dbv.create_dbv_agent()

        # now we should have two sensors and agents
        assert len(sim._Simulator__sensors) == 2
        assert len(sim.agents) == 2

        # collect all the debug visualizer observations for showing later
        dbv_obs = []

        # test scene peeking
        dbv_obs.append(dbv.peek("scene"))

        # test removing the agent/sensor
        dbv.remove_dbv_agent()
        assert len(sim._Simulator__sensors) == 1
        assert len(sim.agents) == 1

        # test switching modes
        dbv.create_dbv_agent()
        assert len(sim._Simulator__sensors) == 2
        assert len(sim.agents) == 2
        assert dbv.agent is not None
        assert dbv.equirect == False
        assert type(dbv.sensor._spec) == habitat_sim.CameraSensorSpec
        dbv.equirect = True
        assert dbv.equirect == True
        assert type(dbv.sensor._spec) == habitat_sim.EquirectangularSensorSpec
        dbv_obs.append(dbv.peek("scene"))

        # optionally show the debug images for local testing
        if show_images:
            for im in dbv_obs:
                im.show()
