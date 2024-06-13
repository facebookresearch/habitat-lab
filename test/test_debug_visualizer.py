#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import numpy as np
import pytest

import habitat_sim
from habitat.sims.habitat_simulator.debug_visualizer import (
    DebugVisualizer,
    stitch_image_matrix,
)
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

        # test the deconstructor
        del dbv

        # test the image matrix stitching utility
        # NOTE: PIL.Image.Image.size is (width, height) while VisualSensor.resolution is (height, width)
        im_width = 200
        im_height = 100
        dbv = DebugVisualizer(sim, resolution=(im_height, im_width))
        # get 10 random images
        obs_cache = [
            dbv.get_observation(np.random.uniform(size=3), np.zeros(3))
            for _ in range(10)
        ]
        im_cache = [obs.get_image() for obs in obs_cache]
        # 3-column stitch of 10 images == 3x4
        stitch_3_col_all = stitch_image_matrix(im_cache, num_col=3)
        assert stitch_3_col_all.get_image().size == (
            im_width * 3,
            im_height * 4,
        )
        # 3-column stitch of 9 images == 3x3
        stitch_3_col_9 = stitch_image_matrix(im_cache[1:], num_col=3)
        assert stitch_3_col_9.get_image().size == (im_width * 3, im_height * 3)
        # 8-column stitch of 10 images == 8x2
        stitch_3_col_9 = stitch_image_matrix(im_cache, num_col=8)
        assert stitch_3_col_9.get_image().size == (im_width * 8, im_height * 2)
        # 8-column stitch of 8 images == 8x1
        stitch_3_col_9 = stitch_image_matrix(im_cache[-8:], num_col=8)
        assert stitch_3_col_9.get_image().size == (im_width * 8, im_height)
        # 8-column stitch of 4 images == 8x1
        stitch_3_col_9 = stitch_image_matrix(im_cache[-4:], num_col=8)
        assert stitch_3_col_9.get_image().size == (im_width * 8, im_height)

        # test assertion that images sizes must match by adding a larger image
        dbv.remove_dbv_agent()
        dbv.create_dbv_agent(resolution=(im_height * 2, im_width * 2))
        larger_image = dbv.get_observation(
            np.random.uniform(size=3), np.zeros(3)
        )
        try:
            stitch_image_matrix(im_cache + [larger_image.get_image()])
        except ValueError as e:
            assert "Image sizes must all match" in (str(e))

        # test assertion that images must be provided
        try:
            stitch_image_matrix([])
        except ValueError as e:
            assert "No images provided" in (str(e))
