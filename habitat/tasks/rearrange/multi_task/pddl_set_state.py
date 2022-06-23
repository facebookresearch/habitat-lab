#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional

import magnum as mn
import numpy as np

from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ART_OBJ_TYPE,
    GOAL_TYPE,
    OBJ_TYPE,
    RIGID_OBJ_TYPE,
    ROBOT_TYPE,
    STATIC_OBJ_TYPE,
    PddlEntity,
    PddlSimInfo,
)
from habitat.tasks.rearrange.utils import rearrange_logger
from habitat.tasks.utils import get_angle


class ArtSampler:
    def __init__(self, value, cmp):
        self.value = value
        self.cmp = cmp

    def is_satisfied(self, cur_value: float, thresh: float) -> bool:
        if self.cmp == "greater":
            return cur_value > self.value
        elif self.cmp == "less":
            return cur_value < self.value
        elif self.cmp == "close":
            return abs(cur_value - self.value) < thresh
        else:
            raise ValueError(f"Unrecognized cmp {self.cmp}")

    def sample(self) -> float:
        return self.value


@dataclass
class PddlRobotState:
    """
    Specifies the configuration of the robot. Only used as a data structure. Not used to set the simulator state.
    """

    holding: Optional[PddlEntity] = None
    should_drop: bool = False
    pos: Optional[Any] = None

    def sub_in(
        self, sub_dict: Dict[PddlEntity, PddlEntity]
    ) -> "PddlRobotState":
        self.holding = sub_dict.get(self.holding, self.holding)
        self.pos = sub_dict.get(self.pos, self.pos)
        return self

    def clone(self) -> "PddlRobotState":
        return PddlRobotState(
            holding=self.holding, should_drop=self.should_drop, pos=self.pos
        )

    def is_true(self, sim_info: PddlSimInfo, robot_entity: PddlEntity) -> bool:
        """
        Returns if the desired robot state is currently true in the simulator state.
        """
        robot_id = sim_info.search_for_entity(robot_entity, ROBOT_TYPE)
        grasp_mgr = sim_info.sim.get_robot_data(robot_id).grasp_mgr

        assert not (self.holding is not None and self.should_drop)

        if self.holding is not None:
            # Robot must be holding desired object.
            obj_idx = sim_info.search_for_entity(self.holding, RIGID_OBJ_TYPE)
            abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
            if grasp_mgr.snap_idx != abs_obj_id:
                return False
        elif self.should_drop and grasp_mgr.snap_idx != None:
            return False

        if isinstance(self.pos, PddlEntity):
            targ_pos = sim_info.get_entity_pos(self.pos)
            robot = sim_info.sim.get_robot_data(robot_id).robot
            dist = np.linalg.norm(robot.base_pos - targ_pos)
            if dist > sim_info.robot_at_thresh:
                return False

        return True

    def set_state(
        self, sim_info: PddlSimInfo, robot_entity: PddlEntity
    ) -> None:
        robot_id = sim_info.search_for_entity(robot_entity, ROBOT_TYPE)
        sim = sim_info.sim
        grasp_mgr = sim.get_robot_data(robot_id).grasp_mgr
        # Set the snapped object information
        if self.should_drop and grasp_mgr.is_grasped:
            grasp_mgr.desnap(True)
        elif self.holding is not None:
            # Swap objects to the desired object.
            obj_idx = sim_info.search_for_entity(self.holding, RIGID_OBJ_TYPE)
            grasp_mgr.desnap(True)
            sim.internal_step(-1)
            grasp_mgr.snap_to_obj(sim.scene_obj_ids[obj_idx])
            sim.internal_step(-1)

        # Set the robot starting position
        if isinstance(self.pos, PddlEntity):
            targ_pos = sim_info.get_entity_pos(self.pos)
            robo_pos = sim_info.sim.safe_snap_point(targ_pos)
            robot = sim.get_robot_data(robot_id).robot
            robot.base_pos = robo_pos

            forward = np.array([0.0, 1.0])
            rel_pos = np.array(targ_pos - robo_pos)[[0, 2]]
            angle = get_angle(forward, rel_pos)
            rearrange_logger.debug(
                f"Setting robot base to {self.pos} at {targ_pos} with angle {angle}."
            )
            robot.base_rot = angle
        elif self.pos is not None:
            raise ValueError(f"Unrecongized set position {self.pos}")


class PddlSetState:
    def __init__(
        self,
        art_states: Dict[PddlEntity, ArtSampler],
        obj_states: Dict[PddlEntity, PddlEntity],
        robot_states: Dict[PddlEntity, PddlRobotState],
    ):
        for k, v in obj_states.items():
            if not isinstance(k, PddlEntity) or not isinstance(v, PddlEntity):
                raise TypeError(f"Unexpected types {obj_states}")

        for k, v in art_states.items():
            if not isinstance(k, PddlEntity) or not isinstance(v, ArtSampler):
                raise TypeError(f"Unexpected types {art_states}")

        for k, v in robot_states.items():
            if not isinstance(k, PddlEntity) or not isinstance(
                v, PddlRobotState
            ):
                raise TypeError(f"Unexpected types {robot_states}")

        self._art_states = art_states
        self._obj_states = obj_states
        self._robot_states = robot_states

    def __repr__(self):
        return f"{self._art_states}, {self._obj_states}, {self._robot_states}"

    def clone(self) -> "PddlSetState":
        return PddlSetState(
            self._art_states,
            self._obj_states,
            {k: v.clone() for k, v in self._robot_states.items()},
        )

    def sub_in(self, sub_dict: Dict[PddlEntity, PddlEntity]) -> "PddlSetState":
        self._robot_states = {
            sub_dict.get(k, k): robot_state.sub_in(sub_dict)
            for k, robot_state in self._robot_states.items()
        }
        self._art_states = {
            sub_dict.get(k, k): v for k, v in self._art_states.items()
        }
        self._obj_states = {
            sub_dict.get(k, k): sub_dict.get(v, v)
            for k, v in self._obj_states.items()
        }
        return self

    def _is_object_inside(
        self, entity: PddlEntity, target: PddlEntity, sim_info: PddlSimInfo
    ):
        if sim_info.check_type_matches(entity, GOAL_TYPE):
            use_receps = sim_info.sim.ep_info["goal_receptacles"]
            obj_idx = sim_info.search_for_entity(entity, GOAL_TYPE)
        elif sim_info.check_type_matches(entity, RIGID_OBJ_TYPE):
            use_receps = sim_info.sim.ep_info["target_receptacles"]
            obj_idx = sim_info.search_for_entity(entity, RIGID_OBJ_TYPE)
        else:
            raise ValueError()

        if not sim_info.check_type_matches(target, ART_OBJ_TYPE):
            raise ValueError()
        check_marker = sim_info.search_for_entity(target, ART_OBJ_TYPE)

        if obj_idx >= len(use_receps):
            rearrange_logger.debug(
                f"Could not find object {entity} in {use_receps}"
            )
            return False

        recep_name, recep_link_id = use_receps[obj_idx]
        if recep_link_id != check_marker.link_id:
            return False
        # if recep_name != check_marker.ao_parent.handle:
        #    return False
        return True

    def is_compatible(self, expr_types) -> bool:
        def type_matches(entity, match_name):
            return entity.expr_type.is_subtype_of(expr_types[match_name])

        for entity, target in self._obj_states.items():

            if not type_matches(entity, OBJ_TYPE):
                return False
            if not type_matches(target, STATIC_OBJ_TYPE):
                return False

            if not (
                type_matches(target, ART_OBJ_TYPE)
                or type_matches(target, GOAL_TYPE)
            ):
                return False

            if entity.expr_type.name == target.expr_type.name:
                return False

        return all(
            type_matches(art_entity, ART_OBJ_TYPE)
            for art_entity in self._art_states
        )

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
                if not self._is_object_inside(entity, target, sim_info):
                    return False
            elif sim_info.check_type_matches(target, GOAL_TYPE):
                obj_idx = sim_info.search_for_entity(entity, RIGID_OBJ_TYPE)
                abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
                cur_pos = rom.get_object_by_id(
                    abs_obj_id
                ).transformation.translation

                targ_idx = sim_info.search_for_entity(target, GOAL_TYPE)
                idxs, pos_targs = sim_info.sim.get_targets()
                targ_pos = pos_targs[list(idxs).index(targ_idx)]

                dist = np.linalg.norm(cur_pos - targ_pos)
                if dist >= sim_info.obj_thresh:
                    return False
            else:
                raise ValueError(
                    f"Got unexpected combination of {entity} and {target}"
                )

        for art_entity, set_art in self._art_states.items():
            if not sim_info.check_type_matches(art_entity, ART_OBJ_TYPE):
                raise ValueError()

            marker = sim_info.search_for_entity(art_entity, ART_OBJ_TYPE)
            prev_art_pos = marker.get_targ_js()
            if not set_art.is_satisfied(prev_art_pos, sim_info.art_thresh):
                return False
        return all(
            robot_state.is_true(sim_info, robot_entity)
            for robot_entity, robot_state in self._robot_states.items()
        )

    def set_state(self, sim_info: PddlSimInfo) -> None:
        """
        Set this state in the simulator. Warning, this steps the simulator.
        """
        sim = sim_info.sim
        for entity, target in self._obj_states.items():
            obj_idx = sim_info.search_for_entity(entity, RIGID_OBJ_TYPE)
            abs_obj_id = sim.scene_obj_ids[obj_idx]

            targ_idx = sim_info.search_for_entity(target, GOAL_TYPE)
            all_targ_idxs, pos_targs = sim.get_targets()
            targ_pos = pos_targs[list(all_targ_idxs).index(targ_idx)]
            set_T = mn.Matrix4.translation(targ_pos)

            # Get the object id corresponding to this name
            rom = sim.get_rigid_object_manager()
            set_obj = rom.get_object_by_id(abs_obj_id)
            set_obj.transformation = set_T
            set_obj.angular_velocity = mn.Vector3.zero_init()
            set_obj.linear_velocity = mn.Vector3.zero_init()
            sim.internal_step(-1)
            set_obj.angular_velocity = mn.Vector3.zero_init()
            set_obj.linear_velocity = mn.Vector3.zero_init()

        for art_entity, set_art in self._art_states.items():
            marker = sim_info.search_for_entity(art_entity, ART_OBJ_TYPE)
            marker.set_targ_js(set_art.sample())
            sim.internal_step(-1)
        for robot_entity, robot_state in self._robot_states.items():
            robot_state.set_state(sim_info, robot_entity)
