#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import magnum as mn
import numpy as np
from gym import spaces

from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction


class HandState(Enum):
    APPROACHING = 0
    RETRACTING = 1


@registry.register_task_action
class HumanoidPickAction(HumanoidJointAction):
    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        HumanoidJointAction.__init__(self, *args, **kwargs)
        self.dbv = None

        self.humanoid_controller = self.lazy_inst_humanoid_controller(
            task, config
        )

        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self._prev_ep_id = None
        self.skill_done = False
        self.hand_state = HandState.APPROACHING

        self.dist_move_per_step = config.dist_move_per_step
        self.dist_to_snap = config.dist_to_snap

        self._init_coord = mn.Vector3(
            0.2, 0.2, 0
        )  # Init coord with respect to the agent root pose.
        self._hand_pose_iter = 0

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if (
            not hasattr(task, "humanoid_controller")
            or task.humanoid_controller is None
        ):
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)
            task.humanoid_controller = humanoid_controller

        self.dbv = DebugVisualizer(self._sim, output_path="")
        return task.humanoid_controller

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "humanoid_pick_action": spaces.Box(
                    shape=(2,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _get_coord_for_idx(self, object_target_idx):
        pick_obj_entity = self._entities[object_target_idx]
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            pick_obj_entity
        )
        return obj_pos

    def get_scene_index_obj(self, object_target_idx):
        pick_obj_entity = self._entities[object_target_idx]
        entity_name = pick_obj_entity.name
        obj_id = self._task.pddl_problem.sim_info.obj_ids[entity_name]
        return self._sim.scene_obj_ids[obj_id]

    def step(self, *args, **kwargs):
        self.skill_done = False
        object_pick_idx = kwargs[
            self._action_arg_prefix + "humanoid_pick_action"
        ][0]
        should_pick = kwargs[self._action_arg_prefix + "humanoid_pick_action"][
            1
        ]

        if object_pick_idx <= 0 or object_pick_idx > len(self._entities):
            return

        object_coord = self._get_coord_for_idx(object_pick_idx)
        init_coord_world = (
            self.humanoid_controller.obj_transform_base.transform_point(
                self._init_coord
            )
        )

        hand_vector = (object_coord - init_coord_world) / np.linalg.norm(
            object_coord - init_coord_world
        )
        max_num_iters = int(
            np.linalg.norm(object_coord - init_coord_world)
            / self.dist_move_per_step
        )

        should_rest = False
        if self.hand_state == HandState.APPROACHING:  # Approaching
            # Only move the hand to object if has to drop or object is not grabbed
            if should_pick == 0 or self.cur_grasp_mgr.snap_idx is None:
                new_hand_coord = (
                    init_coord_world
                    + self._hand_pose_iter
                    * self.dist_move_per_step
                    * hand_vector
                )
                self._hand_pose_iter = min(
                    self._hand_pose_iter + 1, max_num_iters
                )
                dist_hand_obj = np.linalg.norm(object_coord - new_hand_coord)
                if dist_hand_obj < self.dist_to_snap:
                    # snap,
                    self.hand_state = HandState.RETRACTING
                    if should_pick:
                        object_index = self.get_scene_index_obj(
                            object_pick_idx
                        )
                        if self.cur_grasp_mgr.snap_idx is None:
                            self.cur_grasp_mgr.snap_to_obj(
                                object_index,
                            )
                        self._sim.internal_step(-1)
                    else:
                        obj_grabbed = self.cur_grasp_mgr.snap_rigid_obj()
                        self.cur_grasp_mgr.desnap(True)
                        if obj_grabbed is not None:
                            obj_grabbed.transformation = (
                                mn.Matrix4.translation(object_coord)
                            )
            else:
                should_rest = True

        else:  # Retracting
            new_hand_coord = (
                init_coord_world
                + self._hand_pose_iter * self.dist_move_per_step * hand_vector
            )
            self._hand_pose_iter = max(0, self._hand_pose_iter - 1)
            dist_hand_init = np.linalg.norm(new_hand_coord - init_coord_world)
            if dist_hand_init < self.dist_to_snap:
                self.hand_state = HandState.APPROACHING
                self.skill_done = True
                self._hand_pose_iter = 0

        if should_rest:
            self.humanoid_controller.calculate_stop_pose()
        else:
            self.humanoid_controller.calculate_reach_pose(new_hand_coord)

        base_action = self.humanoid_controller.get_pose()
        kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action

        HumanoidJointAction.step(self, *args, **kwargs)
        return

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._prev_ep_id = self._task._episode_id
            self.skill_done = False
        self._hand_pose_iter = 0


@registry.register_task_action
class HumanoidPickObjIdAction(HumanoidPickAction):
    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        HumanoidJointAction.__init__(self, *args, **kwargs)
        self.vdb = None

        self.humanoid_controller = self.lazy_inst_humanoid_controller(
            task, config
        )

        self._task = task
        self._prev_ep_id = None
        self.skill_done = False
        self.hand_state = HandState.APPROACHING

        self.dist_move_per_step = config.dist_move_per_step
        self.dist_to_snap = config.dist_to_snap

        self._init_coord = mn.Vector3(
            0.2, 0.2, 0
        )  # Init coord with respect to the agent root pose.
        self._hand_pose_iter = 0

    def _get_coord_for_idx(self, object_target_idx):
        obj_pos = (
            self._sim.get_rigid_object_manager()
            .get_object_by_id(object_target_idx)
            .translation
        )
        return obj_pos

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "humanoid_pickobj_id": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, **kwargs):
        self.skill_done = False
        object_pick_idx = kwargs[
            self._action_arg_prefix + "humanoid_pick_obj_id"
        ]

        object_coord = self._get_coord_for_idx(object_pick_idx)
        init_coord_world = (
            self.humanoid_controller.obj_transform_base.transform_point(
                self._init_coord
            )
        )

        hand_vector = (object_coord - init_coord_world) / np.linalg.norm(
            object_coord - init_coord_world
        )
        max_num_iters = int(
            np.linalg.norm(object_coord - init_coord_world)
            / self.dist_move_per_step
        )

        should_rest = False
        if self.hand_state == HandState.APPROACHING:  # Approaching
            # Only move the hand to object if has to drop or object is not grabbed
            if self.cur_grasp_mgr.snap_idx is None:
                new_hand_coord = (
                    init_coord_world
                    + self._hand_pose_iter
                    * self.dist_move_per_step
                    * hand_vector
                )
                self._hand_pose_iter = min(
                    self._hand_pose_iter + 1, max_num_iters
                )
                dist_hand_obj = np.linalg.norm(object_coord - new_hand_coord)
                if dist_hand_obj < self.dist_to_snap:
                    # snap,
                    self.hand_state = HandState.RETRACTING
                    object_index = object_pick_idx
                    if self.cur_grasp_mgr.snap_idx is None:
                        self.cur_grasp_mgr.snap_to_obj(
                            object_index,
                        )
                    self._sim.internal_step(-1)

            else:
                should_rest = True

        else:  # Retracting
            new_hand_coord = (
                init_coord_world
                + self._hand_pose_iter * self.dist_move_per_step * hand_vector
            )
            self._hand_pose_iter = max(0, self._hand_pose_iter - 1)
            dist_hand_init = np.linalg.norm(new_hand_coord - init_coord_world)
            if dist_hand_init < self.dist_to_snap:
                self.hand_state = HandState.APPROACHING
                self.skill_done = True
                self._hand_pose_iter = 0

        if should_rest:
            self.humanoid_controller.calculate_stop_pose()
        else:
            self.humanoid_controller.calculate_reach_pose(new_hand_coord)

        base_action = self.humanoid_controller.get_pose()
        kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action

        HumanoidJointAction.step(self, *args, **kwargs)
        return

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._prev_ep_id = self._task._episode_id
            self.skill_done = False
        self._hand_pose_iter = 0
