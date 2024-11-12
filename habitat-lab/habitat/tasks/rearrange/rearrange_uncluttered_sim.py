#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import time
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import magnum as mn
import numpy as np
import numpy.typing as npt

import habitat_sim

# flake8: noqa
from habitat.articulated_agents.robots import FetchRobot, FetchRobotNoWheels
from habitat.config import read_write
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Observations
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.datasets.rearrange.samplers.receptacle import (
    AABBReceptacle,
    find_receptacles,
)
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentData,
    ArticulatedAgentManager,
)
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat.tasks.rearrange.utils import (
    add_perf_timing_func,
    get_rigid_aabb,
    make_render_only,
    rearrange_collision,
    rearrange_logger,
)
from habitat_sim.logging import logger
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import CollisionGroups, JointMotorSettings, MotionType
from habitat_sim.sim import SimulatorBackend
from habitat_sim.utils.common import quat_from_magnum

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_simulator(name="RearrangeUnclutteredSim-v0")
class RearrangeUnclutteredSim(RearrangeSim):
    def __init__(self, config: "DictConfig"):
        super().__init__(config)

    @add_perf_timing_func()
    def reconfigure(self, config: "DictConfig", ep_info: RearrangeEpisode):
        self._handle_to_goal_name = ep_info.info["object_labels"]

        self.ep_info = ep_info
        new_scene = self.prev_scene_id != ep_info.scene_id
        if new_scene:
            self._prev_obj_names = None

        # Only remove and re-add objects if we have a new set of objects.
        ep_info.rigid_objs = sorted(ep_info.rigid_objs, key=lambda x: x[0])
        obj_names = [x[0] for x in ep_info.rigid_objs]
        # Only remove and re-add objects if we have a new set of objects.
        should_add_objects = self._prev_obj_names != obj_names
        self._prev_obj_names = obj_names

        self.agents_mgr.pre_obj_clear()
        self._clear_objects(should_add_objects, new_scene)

        is_hard_reset = new_scene or should_add_objects

        if is_hard_reset:
            with read_write(config):
                config["scene"] = ep_info.scene_id
            t_start = time.time()
            super().reconfigure(config, should_close_on_new_scene=False)
            self.add_perf_timing("super_reconfigure", t_start)
            # The articulated object handles have changed.
            self._start_art_states = {}

        if new_scene:
            self.agents_mgr.on_new_scene()

        self.prev_scene_id = ep_info.scene_id
        self._viz_templates = {}
        self._viz_handle_to_template = {}

        # Set the default articulated object joint state.
        for ao, (set_joint_state, set_T) in self._start_art_states.items():
            ao.clear_joint_states()
            ao.joint_positions = set_joint_state
            if not is_hard_reset:
                # [Andrew Szot 2023-08-22]: If we don't correct for this, some
                # articulated objects may "drift" over time when the scene
                # reset is skipped.
                ao.transformation = set_T

        # Load specified articulated object states from episode config
        self._set_ao_states_from_ep(ep_info)

        self.agents_mgr.post_obj_load_reconfigure()

        self._setup_targets(ep_info)
        self.target_handles_ids = {}
        for k in self._targets.keys():
            self.target_handles_ids[k] = -1

        # add episode clutter objects additional to base scene objects
        if self._load_objs:
            self._add_objs(ep_info, should_add_objects, new_scene)

        self._add_markers(ep_info)
        rom = self.get_rigid_object_manager()

        # auto-sleep rigid objects as optimization
        if self._auto_sleep:
            self._sleep_all_objects()

        self._obj_orig_motion_types = {
            handle: ro.motion_type
            for handle, ro in rom.get_objects_by_handle_substring().items()
        }

        if new_scene:
            self._load_navmesh(ep_info)

        # Get the starting positions of the target objects.
        scene_pos = self.get_scene_pos()
        self.target_start_pos = np.array(
            [
                scene_pos[
                    self._scene_obj_ids.index(
                        rom.get_object_by_handle(t_handle).object_id
                    )
                ]
                for t_handle, _ in self._targets.items()
            ]
        )

        self._draw_bb_objs = [
            rom.get_object_by_handle(obj_handle).object_id
            for obj_handle in self._targets
        ]

        if self.first_setup:
            self.first_setup = False
            self.agents_mgr.first_setup()
            # Capture the starting art states
            self._start_art_states = {
                ao: (ao.joint_positions, ao.transformation)
                for ao in self.art_objs
            }

        if self._should_setup_semantic_ids:
            self._setup_semantic_ids()

    @add_perf_timing_func()
    def _add_objs(
        self,
        ep_info: RearrangeEpisode,
        should_add_objects: bool,
        new_scene: bool,
    ) -> None:
        # Load clutter objects:
        rom = self.get_rigid_object_manager()
        obj_counts: Dict[str, int] = defaultdict(int)

        self._handle_to_object_id = {}
        if should_add_objects:
            self._scene_obj_ids = []

        for i, (obj_handle, transform) in enumerate(ep_info.rigid_objs):
            t_start = time.time()
            target_obj = obj_handle.split(".object_config.json")[0]
            for k in self.target_handles_ids.keys():
                if target_obj in k:
                    if should_add_objects:
                        template = None
                        for obj_path in self._additional_object_paths:
                            template = osp.join(obj_path, obj_handle)
                            if osp.isfile(template):
                                break
                        assert (
                            template is not None
                        ), f"Could not find config file for object {obj_handle}"

                        ro = rom.add_object_by_template_handle(template)
                        print("target_obj: ", target_obj)
                        self.target_handles_ids[k] = ro.object_id
                        print("added ro 1: ", ro)
                    else:
                        ro = rom.get_object_by_id(self.target_handles_ids[k])
                        print("added ro 2: ", ro)
                        # ro = rom.get_object_by_id(self._scene_obj_ids[i])
                    self.add_perf_timing("create_asset", t_start)

                    # The saved matrices need to be flipped when reloading.
                    ro.transformation = mn.Matrix4()
                    ro.angular_velocity = mn.Vector3.zero_init()
                    ro.linear_velocity = mn.Vector3.zero_init()

                    other_obj_handle = (
                        obj_handle.split(".")[0]
                        + f"_:{obj_counts[obj_handle]:04d}"
                    )
                    if self._kinematic_mode:
                        ro.motion_type = (
                            habitat_sim.physics.MotionType.KINEMATIC
                        )
                        ro.collidable = False

                    if should_add_objects:
                        self._scene_obj_ids.append(ro.object_id)

                    rel_idx = self._scene_obj_ids.index(ro.object_id)
                    self._handle_to_object_id[other_obj_handle] = rel_idx

                    if other_obj_handle in self._handle_to_goal_name:
                        ref_handle = self._handle_to_goal_name[
                            other_obj_handle
                        ]
                        self._handle_to_object_id[ref_handle] = rel_idx

                    obj_counts[obj_handle] += 1

        if new_scene:
            self._receptacles = self._create_recep_info(
                ep_info.scene_id, list(self._handle_to_object_id.keys())
            )
            self._receptacles_id = self._receptacles_id_cache[ep_info.scene_id]
            ao_mgr = self.get_articulated_object_manager()
            # Make all articulated objects (including the robots) kinematic
            for aoi_handle in ao_mgr.get_object_handles():
                ao = ao_mgr.get_object_by_handle(aoi_handle)
                if self._kinematic_mode:
                    ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                    # remove any existing motors when converting to kinematic AO
                    for motor_id in ao.existing_joint_motor_ids:
                        ao.remove_joint_motor(motor_id)
                self.art_objs.append(ao)
