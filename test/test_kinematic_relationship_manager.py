#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import pytest

from habitat.sims.habitat_simulator.kinematic_relationship_manager import (
    KinematicRelationshipManager,
)
from habitat_sim import Simulator
from habitat_sim.utils.settings import default_sim_settings, make_cfg


@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_kinematic_relationship_manager():
    """
    Test managing some kinematic states within ReplicaCAD "apt_0".
    """

    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)

    with Simulator(hab_cfg) as sim:
        # need to settle ReplicaCAD so items rest on surfaces
        sim.step_physics(2.0)
        # construct the krm and initialize relationships
        krm = KinematicRelationshipManager(sim)

        # TODO: figure out the cyclic relationships here
        krm.initialize_relationship_graph()

        krm.relationship_graph.get_human_readable_relationship_forest(
            sim, do_print=True
        )
