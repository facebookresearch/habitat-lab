#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Utilities built on top of Habitat-lab


def get_agent_art_obj(sim, agent_idx):
    assert agent_idx is not None
    art_obj = sim.agents_mgr[agent_idx].articulated_agent.sim_obj
    return art_obj


def get_agent_art_obj_transform(sim, agent_idx):
    assert agent_idx is not None
    art_obj = sim.agents_mgr[agent_idx].articulated_agent.sim_obj
    return art_obj.transformation


def get_grasped_objects_idxs(sim, agent_idx_to_skip=None):
    agents_mgr = sim.agents_mgr

    grasped_objects_idxs = []
    for agent_idx in range(len(agents_mgr._all_agent_data)):
        if agent_idx == agent_idx_to_skip:
            continue
        grasp_mgr = agents_mgr._all_agent_data[agent_idx].grasp_mgr
        if grasp_mgr.is_grasped:
            grasped_objects_idxs.append(
                sim.scene_obj_ids.index(grasp_mgr.snap_idx)
            )

    return grasped_objects_idxs
