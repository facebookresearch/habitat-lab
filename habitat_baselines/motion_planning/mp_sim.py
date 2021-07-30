#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Optional

from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import get_aabb, rearrange_collision
from habitat_baselines.motion_planning.robot_target import ObjectGraspTarget
from habitat_sim.physics import CollisionGroups, MotionType


class MpSim(ABC):
    """
    The abstract simulator interface for the motion planner.
    """

    def __init__(self, sim: RearrangeSim):
        self._sim = sim
        self._ik = self._sim.ik_helper

    def setup(self, use_prev):
        pass

    def should_ignore_first_collisions(self):
        return False

    @abstractmethod
    def set_targ_obj_idx(self, targ_obj_idx):
        pass

    @abstractmethod
    def unset_targ_obj_idx(self, targ_obj_idx):
        pass

    @abstractmethod
    def get_robot_transform(self):
        """
        Returns the robot to world transformation matrix.
        """

    @abstractmethod
    def get_collisions(self, count_obj_colls, ignore_names, verbose):
        """
        Returns a list of pairs that collided where each element in the pair is
        of the form:
            {
            "name": "body name",
            "link": "link name",
            }
        """

    @abstractmethod
    def capture_state(self):
        pass

    @abstractmethod
    def get_arm_pos(self):
        pass

    @abstractmethod
    def set_arm_pos(self, joint_pos):
        pass

    @abstractmethod
    def set_position(self, pos, obj_id):
        pass

    @abstractmethod
    def micro_step(self):
        pass

    @abstractmethod
    def add_sphere(self, radius, color=None):
        pass

    @abstractmethod
    def get_ee_pos(self):
        """
        Gets the end-effector position in GLOBAL coordinates
        """

    @abstractmethod
    def remove_object(self, obj_id):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def render(self):
        """
        Renders the current state of the simulator.
        """

    @abstractmethod
    def start_mp(self):
        pass

    @abstractmethod
    def end_mp(self):
        pass

    @abstractmethod
    def get_obj_info(self, obj_idx) -> ObjectGraspTarget:
        """
        Returns information about an object for the grasp planner
        """


class HabMpSim(MpSim):
    def get_collisions(
        self,
        count_obj_colls: bool,
        ignore_names: Optional[List[str]],
        verbose: bool,
    ):
        return rearrange_collision(
            self._sim,
            count_obj_colls,
            ignore_names=ignore_names,
            verbose=verbose,
            get_extra_coll_data=True,
        )

    @property
    def _snap_idx(self):
        return self._sim.grasp_mgr.snap_idx

    def capture_state(self):
        env_state = self._sim.capture_state()
        return env_state

    def get_ee_pos(self):
        return self._sim.robot.ee_transform.translation

    def set_state(self, state):
        if self._snap_idx is not None:
            # Auto-snap the held object to the robot's hand.
            local_idx = self._sim.scene_obj_ids.index(self._snap_idx)
            state["static_T"][local_idx] = self._sim.robot.ee_transform
        self._sim.set_state(state)

    def set_arm_pos(self, joint_pos):
        self._sim.robot.arm_joint_pos = joint_pos

    def get_robot_transform(self):
        return self._sim.robot.base_transformation

    def get_obj_info(self, obj_idx) -> ObjectGraspTarget:
        return ObjectGraspTarget(
            bb=get_aabb(obj_idx, self._sim),
            translation=self._sim.get_transformation(obj_idx),
        )

    def set_position(self, pos, obj_id):
        self._sim.set_translation(pos, obj_id)

    def get_arm_pos(self):
        return self._sim.robot.arm_joint_pos

    def micro_step(self):
        # self._sim.perform_discrete_collision_detection()
        self._sim.internal_step(-1)

    def add_sphere(self, radius, color=None):
        sphere_id = self._sim.draw_sphere(radius)

        rigid_obj = self._sim.get_rigid_object_manager().get_object_by_id(
            sphere_id
        )
        rigid_obj.override_collision_group(CollisionGroups.UserGroup7)
        return sphere_id

    def remove_object(self, obj_id):
        self._sim.remove_object(obj_id)

    def set_targ_obj_idx(self, targ_obj_idx):
        if targ_obj_idx is not None:
            self._sim.override_collision_group(targ_obj_idx, 128)

    def unset_targ_obj_idx(self, targ_obj_idx):
        if targ_obj_idx is not None:
            self._sim.override_collision_group(targ_obj_idx, 8)

    def render(self):
        obs = self._sim.step(0)
        if "robot_third_rgb" not in obs:
            raise ValueError("No render camera")
        pic = obs["robot_third_rgb"]
        if pic.shape[-1] > 3:
            # Skip the depth part.
            pic = pic[:, :, :3]
        return pic

    def start_mp(self):
        self.prev_motion_types = {}
        self.hold_obj = self._snap_idx
        if self.hold_obj is not None:
            self._sim.grasp_mgr.desnap(force=True)
            self._sim.do_grab_using_constraint = False
            self._sim.grasp_mgr.snap_to_obj(self.hold_obj)

        # Set everything to STATIC
        for obj_id in self._sim.scene_obj_ids:
            self.prev_motion_types[obj_id] = self._sim.get_object_motion_type(
                obj_id
            )
            if obj_id == self._snap_idx:
                pass
                # self._sim.set_object_motion_type(MotionType.KINEMATIC, obj_id)
            else:
                self._sim.set_object_motion_type(MotionType.STATIC, obj_id)

    def end_mp(self):
        # Set everything to how it was
        for obj_id, mt in self.prev_motion_types.items():
            self._sim.set_object_motion_type(mt, obj_id)

        if self.hold_obj is not None:
            self._sim.grasp_mgr.desnap(force=True)
            self._sim.do_grab_using_constraint = True
            self._sim.grasp_mgr.snap_to_obj(self.hold_obj)
