#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np

from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_sim.physics import (
    CollisionGroupHelper,
    CollisionGroups,
    MotionType,
)


class GuiRobotController(GuiController):
    """Controller for robot agent."""

    def __init__(
        self,
        agent_idx,
        is_multi_agent,
        gui_input,
        articulated_agent,
        num_actions: int,
        base_vel_action_idx: int,
        num_base_vel_actions: int,
        turn_scale: float,
    ):
        super().__init__(agent_idx, is_multi_agent, gui_input)
        self._articulated_agent = articulated_agent
        self._hint_walk_dir = None
        self._hint_distance_multiplier = None
        self._cam_yaw = None
        self._hint_target_dir = None
        self._base_vel_action_idx = base_vel_action_idx
        self._num_base_vel_actions = num_base_vel_actions
        self._turn_scale = turn_scale

        self._actions = np.zeros((num_actions,))

    def set_act_hints(
        self,
        walk_dir,
        distance_multiplier,
        grasp_obj_idx,
        do_drop,
        cam_yaw=None,
        throw_vel=None,
        reach_pos=None,
        hand_idx=None,
        target_dir=None,
    ):
        assert (
            throw_vel is None or do_drop is None
        ), "You can not set throw_velocity and drop_position at the same time"
        self._hint_walk_dir = walk_dir
        self._hint_distance_multiplier = distance_multiplier

        # grasping, dropping, throwing, and reaching aren't supported yet in GuiRobotController
        assert grasp_obj_idx is None
        assert do_drop is None
        assert throw_vel is None
        assert reach_pos is None
        assert hand_idx is None

        self._cam_yaw = cam_yaw
        self._hint_target_dir = target_dir

    def angle_from_sim_obj_forward_dir_to_target_yaw(
        self, sim_obj, target_yaw
    ):
        dir_a = sim_obj.rotation.transform_vector_normalized(
            mn.Vector3(1, 0, 0)
        )
        dir_b = mn.Vector3(
            mn.math.cos(mn.Rad(target_yaw)), 0, mn.math.sin(mn.Rad(target_yaw))
        )
        return self.angle_from_dir_a_to_b(dir_a, dir_b)

    # todo: find a better home for this utility
    def angle_from_dir_a_to_b(self, a: mn.Vector3, b: mn.Vector3):
        assert a.is_normalized() and b.is_normalized()

        # Dot product of vectors a and b
        dot_product = mn.math.dot(a, b)

        # Ensure the dot product does not exceed the range [-1, 1]
        dot_product = max(min(dot_product, 1.0), -1.0)

        # Angle from a to b
        angle = float(mn.math.acos(dot_product))

        # Determinant (2D cross product) of vectors a and b to find direction
        det = a[0] * b[2] - a[2] * b[0]

        # Adjust the angle based on the direction
        return angle if det >= 0 else -angle

    def act(self, obs, env):
        if self._is_multi_agent:
            agent_k = f"agent_{self._agent_idx}_"
        else:
            agent_k = ""
        base_k = f"{agent_k}base_vel"
        base_name = f"{agent_k}base_velocity"
        ac_spaces = env.action_space.spaces

        assert base_name in ac_spaces
        base_action_space = ac_spaces[base_name][base_k]
        base_action = np.zeros(base_action_space.shape[0])

        gui_input = self._gui_input

        # Add 180 degrees due to our camera convention. See camera_helper.py _get_eye_and_lookat. Our camera yaw is used to offset the camera eye pos away from the lookat pos, so the resulting look direction yaw (from eye to lookat) is actually 180 degrees away from this yaw.
        turn_angle = self.angle_from_sim_obj_forward_dir_to_target_yaw(
            self._articulated_agent.sim_obj,
            self._cam_yaw + float(mn.Rad(mn.Deg(180))),
        )

        # sloppy: read gui_input directly instead of using _hint_walk_dir
        if gui_input.get_key(KeyCode.W):
            # walk forward in the camera yaw direction
            base_action[0] += 1
        if gui_input.get_key(KeyCode.S):
            # walk forward in the opposite to camera yaw direction
            base_action[0] -= 1

        # Use anv vel action to turn to face cam yaw. Note that later, this action will get clamped to (-1, 1), so, in the next env step, we may not turn as much as computed here. This can cause the Spot facing direction to slightly lag behind the camera yaw as yaw changes, which is fine.
        base_action[1] = -turn_angle * self._turn_scale

        assert len(base_action) == self._num_base_vel_actions
        self._actions[
            self._base_vel_action_idx : self._base_vel_action_idx
            + self._num_base_vel_actions
        ] = base_action

        return self._actions


class GuiHumanoidController(GuiController):
    """
    A controller for animated humanoid agents.
    Supports walking, picking and placing objects, along with the associated animations.
    """

    def __init__(
        self,
        agent_idx,
        is_multi_agent,
        gui_input,
        env,
        walk_pose_path,
        lin_speed,
        ang_speed,
        recorder,
    ):
        super().__init__(agent_idx, is_multi_agent, gui_input)
        humanoid_controller = HumanoidRearrangeController(walk_pose_path)
        humanoid_controller.set_framerate_for_linspeed(
            lin_speed, ang_speed, env._sim.ctrl_freq
        )
        self._humanoid_controller = humanoid_controller
        self._env = env
        self._hint_walk_dir = None
        self._hint_distance_multiplier = None
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None
        self._hint_throw_vel = (
            None  # The velocity vector at which the object will be thrown
        )
        self._hint_reach_pos = None  # The location at which to pick the object. If None, no object should be picked
        self._is_picking_reach_pos = None  # Same as _hint_reach_pos but is set to None as soon as the humanoid reaches the object
        self._cam_yaw = 0
        self._saved_object_rotation = None
        self._recorder = recorder
        self._obj_to_grasp = None
        self._thrown_object_collision_group = CollisionGroups.UserGroup7
        self._last_object_thrown_info = None

    def set_humanoid_controller(self, humanoid_controller):
        self._humanoid_controller = humanoid_controller

    def get_articulated_agent(self):
        return self._env._sim.agents_mgr[self._agent_idx].articulated_agent

    def get_base_translation(self):
        return self._humanoid_controller.obj_transform_base.translation

    def on_environment_reset(self):
        super().on_environment_reset()
        base_trans = self.get_articulated_agent().base_transformation
        self._humanoid_controller.reset(base_trans)
        self._hint_walk_dir = None
        self._hint_distance_multiplier = None
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None
        self._cam_yaw = 0
        self._hint_throw_vel = None
        self._last_object_thrown_info = None
        self._is_picking_reach_pos = None
        self._obj_to_grasp = None
        self._hint_target_dir = None
        # Disable collision between thrown object and the agents.
        # Both agents (robot and humanoid) have the collision group Robot.
        CollisionGroupHelper.set_mask_for_group(
            self._thrown_object_collision_group, ~CollisionGroups.Robot
        )
        assert not self.is_grasped

    def get_random_joint_action(self):
        # Add random noise to human arms but keep global transform
        (
            joint_trans,
            root_trans,
        ) = self.get_articulated_agent().get_joint_transform()
        # Divide joint_trans by 4 since joint_trans has flattened quaternions
        # and the dimension of each quaternion is 4
        num_joints = len(joint_trans) // 4
        root_trans = np.array(root_trans)
        index_arms_start = 10
        joint_trans_quat = [
            mn.Quaternion(
                mn.Vector3(joint_trans[(4 * index) : (4 * index + 3)]),
                joint_trans[4 * index + 3],
            )
            for index in range(num_joints)
        ]
        rotated_joints_quat = []
        for index, joint_quat in enumerate(joint_trans_quat):
            random_vec = np.random.rand(3)
            # We allow for maximum 10 angles per step
            random_angle = np.random.rand() * 10
            rotation_quat = mn.Quaternion.rotation(
                mn.Rad(random_angle), mn.Vector3(random_vec).normalized()
            )
            if index > index_arms_start:
                joint_quat *= rotation_quat
            rotated_joints_quat.append(joint_quat)
        joint_trans = np.concatenate(
            [
                np.array(list(quat.vector) + [quat.scalar])
                for quat in rotated_joints_quat
            ]
        )
        humanoidjoint_action = np.concatenate(
            [joint_trans.reshape(-1), root_trans.transpose().reshape(-1)]
        )
        return humanoidjoint_action

    def set_act_hints(
        self,
        walk_dir,
        distance_multiplier,
        grasp_obj_idx,
        do_drop,
        cam_yaw=None,
        throw_vel=None,
        reach_pos=None,
        hand_idx=None,
        target_dir=None,
    ):
        assert (
            throw_vel is None or do_drop is None
        ), "You can not set throw_velocity and drop_position at the same time"
        self._hint_walk_dir = walk_dir
        self._hint_distance_multiplier = distance_multiplier
        self._hint_grasp_obj_idx = grasp_obj_idx
        self._hint_drop_pos = do_drop
        self._cam_yaw = cam_yaw
        self._hint_throw_vel = throw_vel
        self._hint_reach_pos = reach_pos
        self._hint_target_dir = target_dir
        self._hand_idx = hand_idx

    def _get_grasp_mgr(self):
        agents_mgr = self._env._sim.agents_mgr
        grasp_mgr = agents_mgr._all_agent_data[self._agent_idx].grasp_mgr
        return grasp_mgr

    @property
    def is_grasped(self):
        return self._get_grasp_mgr().is_grasped

    def _update_grasp(self, grasp_object_id, drop_pos, speed):
        if grasp_object_id is not None:
            assert not self.is_grasped

            sim = self._env.task._sim
            rigid_obj = sim.get_rigid_object_manager().get_object_by_id(
                grasp_object_id
            )
            self._saved_object_rotation = rigid_obj.rotation

            self._get_grasp_mgr().snap_to_obj(grasp_object_id)

            self._recorder.record("grasp_object_id", grasp_object_id)

        elif drop_pos is not None:
            assert self.is_grasped
            grasp_object_id = self._get_grasp_mgr().snap_idx
            self._get_grasp_mgr().desnap()

            # teleport to requested drop_pos
            sim = self._env.task._sim
            rigid_obj = sim.get_rigid_object_manager().get_object_by_id(
                grasp_object_id
            )
            rigid_obj.translation = drop_pos
            rigid_obj.rotation = self._saved_object_rotation
            self._saved_object_rotation = None

            self._recorder.record("drop_pos", drop_pos)

        elif speed is not None:
            grasp_mgr = self._get_grasp_mgr()
            grasp_object_id = grasp_mgr.snap_idx
            grasp_mgr.desnap()
            sim = self._env.task._sim
            rigid_obj = sim.get_rigid_object_manager().get_object_by_id(
                grasp_object_id
            )
            rigid_obj.motion_type = MotionType.DYNAMIC
            rigid_obj.collidable = True
            rigid_obj.override_collision_group(
                self._thrown_object_collision_group
            )
            rigid_obj.linear_velocity = speed

            obj_bb = rigid_obj.aabb
            self._last_object_thrown_info = (
                rigid_obj,
                max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
            )

        if self._last_object_thrown_info is not None:
            grasp_mgr = self._get_grasp_mgr()

            # when the thrown object leaves the hand, update the collisiongroups
            rigid_obj = self._last_object_thrown_info[0]
            ee_pos = (
                self.get_articulated_agent()
                .ee_transform(grasp_mgr.ee_index)
                .translation
            )
            dist = np.linalg.norm(ee_pos - rigid_obj.translation)
            if dist >= self._last_object_thrown_info[1]:
                rigid_obj.override_collision_group(CollisionGroups.Default)
                self._last_object_thrown_info = None

    def update_pick_pose(self):
        hand_pose = self._is_picking_reach_pos
        self._is_picking_reach_pos = None
        if self._obj_to_grasp is not None:
            self._get_grasp_mgr().snap_to_obj(self._obj_to_grasp)
        self._obj_to_grasp = None
        return hand_pose

    def act(self, obs, env):
        self._update_grasp(
            self._hint_grasp_obj_idx,
            self._hint_drop_pos,
            self._hint_throw_vel,
        )
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None
        self._hint_throw_vel = None

        gui_input = self._gui_input

        humancontroller_base_user_input = np.zeros(3)
        # sloppy: read gui_input directly instead of using _hint_walk_dir
        if gui_input.get_key(KeyCode.W):
            # walk forward in the camera yaw direction
            humancontroller_base_user_input[0] += 1
        if gui_input.get_key(KeyCode.S):
            # walk forward in the opposite to camera yaw direction
            humancontroller_base_user_input[0] -= 1

        if self._hint_walk_dir:
            humancontroller_base_user_input[0] += self._hint_walk_dir.x
            humancontroller_base_user_input[2] += self._hint_walk_dir.z

            self._recorder.record("hint_walk_dir", self._hint_walk_dir)

        else:
            self._recorder.record("cam_yaw", self._cam_yaw)
            self._recorder.record(
                "walk_forward_back", humancontroller_base_user_input[0]
            )

            rot_y_rad = -self._cam_yaw + np.pi
            rotation = mn.Quaternion.rotation(
                mn.Rad(rot_y_rad),
                mn.Vector3(0, 1, 0),
            )
            humancontroller_base_user_input = np.array(
                rotation.transform_vector(
                    mn.Vector3(humancontroller_base_user_input)
                )
            )

        self._recorder.record(
            "base_user_input", humancontroller_base_user_input
        )

        relative_pos = mn.Vector3(humancontroller_base_user_input)

        base_offset = self.get_articulated_agent().params.base_offset
        # base_offset is basically the offset from the humanoid's root (often
        # located near its pelvis) to the humanoid's feet (where it should
        # snap to the navmesh), for example (0, -0.9, 0).
        prev_query_pos = (
            self._humanoid_controller.obj_transform_base.translation
            + base_offset
        )

        # 1.0 is the default value indicating moving in the relative_pos direction
        # 0.0 indicates no movement but possibly a rotation on the spot
        distance_multiplier = (
            1.0
            if self._hint_distance_multiplier is None
            else self._hint_distance_multiplier
        )

        # Compute the walk pose.
        # Note: This doesn't use the same 'calculate_walk_pose' method as other lab components.
        self._humanoid_controller.calculate_walk_pose_directional(
            relative_pos, distance_multiplier, self._hint_target_dir
        )

        # calculate_walk_pose has updated obj_transform_base.translation with
        # desired motion, but this should be filtered (restricted to navmesh).
        target_query_pos = (
            self._humanoid_controller.obj_transform_base.translation
            + base_offset
        )
        filtered_query_pos = self._env._sim.step_filter(
            prev_query_pos, target_query_pos
        )
        # fixup is the difference between the movement allowed by step_filter
        # and the requested base movement.
        fixup = filtered_query_pos - target_query_pos

        # TODO: Get Y offset from source data.
        navmesh_to_floor_y_fixup = -0.17
        fixup.y += navmesh_to_floor_y_fixup

        self._humanoid_controller.obj_transform_base.translation += fixup

        # TODO: remove the joint angles overwrite here
        if self._hint_reach_pos:
            self._is_picking_reach_pos = self._hint_reach_pos

        if self._is_picking_reach_pos:
            reach_pos = self.update_pick_pose()
            hand_index = 0 if self._hand_idx is None else self._hand_idx
            self._humanoid_controller.calculate_reach_pose(
                reach_pos, index_hand=hand_index
            )

        humanoidjoint_action = np.array(
            self._humanoid_controller.get_pose(), dtype=np.float32
        )

        return humanoidjoint_action
