#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import magnum as mn
import numpy as np

from habitat import Config
from habitat.core.dataset import Episode
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType,
    PddlEntity,
    PddlSimInfo,
)
from habitat.tasks.rearrange.multi_task.task_creator_utils import (
    create_task_object,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger

# Hardcoded pddl types needed for setting simulator states.
ROBOT_TYPE = "robot_type"
STATIC_OBJ_TYPE = "static_obj"
ART_OBJ_TYPE = "art_obj_type"
OBJ_TYPE = "obj_type"
CAB_TYPE = "cab_type"
FRIDGE_TYPE = "fridge_type"
GOAL_TYPE = "goal_type"
RIGID_OBJ_TYPE = "rigid_obj_type"


@dataclass
class PddlRobotState:
    """
    Specifies the configuration of the robot. Only used as a data structure. Not used to set the simulator state.
    """

    holding: Optional[PddlEntity] = None
    should_drop: bool = False
    pos: Optional[Any] = None

    def bind(self, arg_k, arg_v):
        for k, v in zip(arg_k, arg_v):
            if self.holding is not None:
                self.holding = self.holding.replace(k, v)
            if self.pos is not None:
                self.pos = self.pos.replace(k, v)

    def is_true(self, sim_info: PddlSimInfo, robot_entity: PddlEntity) -> bool:
        """
        Returns if the desired robot state is currently true in the simulator state.
        """
        robot_id = sim_info.search_for_entity(robot_entity)
        grasp_mgr = sim_info.sim.get_robot_data(robot_id).grasp_mgr

        assert not (self.holding is not None and self.should_drop)

        if self.holding is not None:
            # Robot must be holding desired object.
            obj_idx = sim_info.search_for_entity(self.holding)
            abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
            if grasp_mgr.snap_idx != abs_obj_id:
                return False
        elif self.should_drop and grasp_mgr.snap_idx != None:
            return False

        return True

    def set_state(
        self, sim_info: PddlSimInfo, robot_entity: PddlEntity
    ) -> None:
        robot_id = sim_info.search_for_entity(robot_entity)
        sim = sim_info.sim
        grasp_mgr = sim.get_robot_data(robot_id).grasp_mgr
        # Set the snapped object information
        if self.should_drop and grasp_mgr.is_grasped:
            grasp_mgr.desnap(True)
        elif self.holding is not None:
            # Swap objects to the desired object.
            obj_idx = sim_info.search_for_entity(self.holding)
            grasp_mgr.desnap(True)
            sim.internal_step(-1)
            grasp_mgr.snap_to_obj(sim.scene_obj_ids[obj_idx])
            sim.internal_step(-1)

        # Set the robot starting position
        if self.pos == "rnd":
            sim.set_robot_base_to_random_point(agent_idx=robot_id)


class PddlSetState:
    """
    A partially specified state of the simulator. First this object needs to be
    bound to a specific set of arguments specifying scene entities
    (`self.bind`). After, you can query this object to get if the specified
    scene state is satifisfied and set everything specified.
    """

    def __init__(
        self,
        art_states: Dict[PddlEntity, ArtSampler],
        obj_states: Dict[PddlEntity, PddlEntity],
        robot_states: Dict[PddlEntity, PddlRobotState],
    ):
        self._art_states = art_states
        self._obj_states = obj_states
        self._robot_states = robot_states

    def clone(self) -> PddlSetState:
        return PddlSetState(
            self._art_states,
            self.obj_states,
            {k: v.clone() for k, v in self._robot_states.items()},
        )

    def bind(self, arg_k: List[str], arg_v: List[str]) -> None:
        """
        Defines a state in the environment grounded in scene entities.
        :param arg_k: The names of the environment parameters to set.
        :param arg_v: The values of the environment parameters to set.
        """

        def list_replace(l, k, v):
            new_l = {}
            for l_k, l_v in l.items():
                if isinstance(l_k, str):
                    l_k = l_k.replace(k, v)
                if isinstance(l_v, str):
                    l_v = l_v.replace(k, v)
                new_l[l_k] = l_v
            return new_l

        for k, v in zip(arg_k, arg_v):
            self.art_states = list_replace(self.art_states, k, v)
            self.obj_states = list_replace(self.obj_states, k, v)
            if "catch_ids" in self.load_config:
                self.load_config["catch_ids"] = self.load_config[
                    "catch_ids"
                ].replace(k, v)

        for robot_state in self.robot_states:
            robot_state.bind(arg_k, arg_v)
        self._set_args = arg_v

    def _is_object_inside(
        self, entity: PddlEntity, target: PddlEntity, sim_info: PddlSimInfo
    ):
        if sim_info.check_type_matches(entity, GOAL_TYPE):
            use_receps = sim_info.sim.ep_info["goal_receptacles"]
        elif sim_info.check_type_matches(entity, RIGID_OBJ_TYPE):
            use_receps = sim_info.sim.ep_info["target_receptacles"]
            obj_name = list(sim_info.sim.get_targets()[0]).index(int(obj_name))
        else:
            raise ValueError()

        obj_idx = int(obj_name)

        if not sim_info.check_type_matches(target, ART_OBJ_TYPE):
            raise ValueError()
        check_marker = sim_info.search_for_entity(target)

        if obj_idx >= len(use_receps):
            rearrange_logger.debug(
                f"Could not find object {obj_name} in {use_receps}"
            )
            return False

        recep_name, recep_link_id = use_receps[obj_idx]
        if recep_link_id != check_marker.link_id:
            return False
        # if recep_name != check_marker.ao_parent.handle:
        #    return False
        return True

    def is_true(
        self,
        sim_info: PddlSimInfo,
    ) -> bool:
        """
        Returns True if the grounded state is present in the current simulator state.
        Throws exception if the arguments are not compatible.
        """

        # Check object states.
        rom = sim_info.sim.get_rigid_object_manager()
        for entity, target in self._obj_states.items():
            if not sim_info.check_type_matches(entity, OBJ_TYPE):
                raise ValueError()
            if not sim_info.check_type_matches(target, STATIC_OBJ_TYPE):
                raise ValueError()

            if sim_info.check_type_matches(target, ART_OBJ_TYPE):
                # object is rigid and target is receptacle, we are checking if
                # an object is inside of a receptacle.
                if self._is_object_inside(entity, target, sim_info):
                    continue
                else:
                    return False
            elif sim_info.check_type_matches(target, OBJ_TYPE):
                obj_idx = sim_info.search_for_entity(entity)
                abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
                cur_pos = rom.get_object_by_id(
                    abs_obj_id
                ).transformation.translation

                targ_idx = sim_info.search_for_entity(entity)
                idxs, pos_targs = sim_info.sim.get_targets()
                targ_pos = pos_targs[list(idxs).index(targ_idx)]

                dist = np.linalg.norm(cur_pos - targ_pos)
                if dist >= sim_info.obj_thresh:
                    return False
            else:
                raise ValueError()

        for art_entity, set_art in self._art_states.items():
            if not sim_info.check_type_matches(art_entity, ART_OBJ_TYPE):
                raise ValueError()

            marker = sim_info.search_for_entity(art_entity)
            prev_art_pos = marker.get_targ_js()
            if not set_art.is_satisfied(prev_art_pos):
                return False

        for robot_entity, robot_state in self._robot_states.items():
            if not robot_state.is_true(sim_info, robot_entity):
                return False

        return True

    def set_state(self, sim_info: PddlSimInfo) -> None:
        """
        Set this state in the simulator. Warning, this steps the simulator.
        """
        sim = sim_info.sim
        for entity, target in self._obj_states.items():
            obj_idx = sim_info.search_for_entity(entity)
            abs_obj_id = sim.scene_obj_ids[obj_idx]

            targ_idx = sim_info.search_for_entity(target)
            all_targ_idxs, pos_targs = sim.get_targets()
            targ_pos = pos_targs[list(all_targ_idxs).index(targ_idx)]
            set_T = mn.Matrix4.translation(targ_pos)

            # Get the object id corresponding to this name
            rom = sim.get_rigid_object_manager()
            set_obj = rom.get_object_by_id(abs_obj_id)
            set_obj.transformation = set_T

        for art_entity, set_art in self._art_states.items():
            marker = sim_info.search_for_entity(art_entity)
            marker.set_targ_js(set_art.sample())
            sim.internal_step(-1)
        for robot_entity, robot_state in self._robot_states.items():
            robot_state.set_state(sim_info, robot_entity)


class ArtSampler:
    def __init__(self, value, cmp, thresh=0.05):
        self.value = value
        self.cmp = cmp
        self._thresh = thresh

    def is_satisfied(self, cur_value: float) -> bool:
        if self.cmp == "greater":
            return cur_value > self.value
        elif self.cmp == "less":
            return cur_value < self.value
        elif self.cmp == "close":
            return abs(cur_value - self.value) < self._thresh
        else:
            raise ValueError(f"Unrecognized cmp {self.cmp}")

    def sample(self) -> float:
        return self.value
