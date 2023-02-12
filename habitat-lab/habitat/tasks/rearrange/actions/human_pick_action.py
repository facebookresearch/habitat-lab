# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import BaseVelAction, HumanJointAction
from habitat.tasks.rearrange.utils import get_robot_spawns
from habitat.tasks.utils import get_angle

import magnum as mn
import human_controllers.amass_human_controller as amass_human_controller
import quaternion



@registry.register_task_action
class HumanPickAction(HumanJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene.
    """
     
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = task
        self.human_controller = None
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )

        self._prev_ep_id = None
        self._targets = {}
        self.gen = 0
        self.curr_ind_map = {'cont': 0}
        self.counter = 0

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "human_pick_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.counter = 0
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            self._targets[nav_to_target_idx] = np.array(obj_pos)
        return self._targets[nav_to_target_idx]

    def step(self, *args, is_last_action, **kwargs):
        pick_target_idx = kwargs[
            self._action_arg_prefix + "human_pick_action"
        ]
        if pick_target_idx <= 0 or pick_target_idx > len(
            self._poss_entities
        ):
            if is_last_action:
                return self._sim.step(HabitatSimActions.base_velocity)
            else:
                return {}
        
        pick_target_idx = int(pick_target_idx[0]) - 1

        obj_targ_pos = self._get_target_for_idx(
            pick_target_idx
        )
        
        if self.counter == 0:
            self.init_root_pos = self.human_controller.root_pos
            self.motion_vec = obj_targ_pos - self.human_controller.root_pos
            
            # We will need to rotate Z
            euler_angle = quaternion.as_euler_angles(quaternion.from_rotation_matrix(self.human_controller.root_rot))
            new_euler_angle = euler_angle.copy()
            new_euler_angle[:2] = 0
            new_euler_angle[-1] = -new_euler_angle[-1]
            self.new_matrix_transform = quaternion.as_rotation_matrix(quaternion.from_euler_angles(new_euler_angle))
            self.inv_matrix_transform = quaternion.as_rotation_matrix(quaternion.from_euler_angles(-new_euler_angle))
            
            # self.motion_vec = self.new_matrix_transform * mn.Vector3(self.motion_vec)
            
        motion_amount = np.linspace(0.2, 1, 10)[self.counter]
        # breakpoint()
        curr_pos = self.init_root_pos + motion_amount * (self.motion_vec @ self.new_matrix_transform) 

        sim = self._sim
        if 'next_pick' not in sim.viz_ids:
            sim.viz_ids[f'next_pick'] = sim.visualize_position(
                curr_pos
            )
        else:
            sim.viz_ids[f'next_pick'] = sim.visualize_position(
                curr_pos, sim.viz_ids[f'next_pick']
            )

        # breakpoint()
        self.counter += 1
        self.counter = min(self.counter, 9)

        # print(self.human_controller.base_pos, self.cur_human.base_pos)
        pos_rel = curr_pos - self.human_controller.root_pos
        new_pos, new_trans = self.human_controller.reach(mn.Vector3(pos_rel))
        breakpoint()
        new_trans.rotation = new_trans.rotation @ self.inv_matrix_transform 
        # breakpoint()
        # print("DISTANCE AND MOTION", dist_to_final_nav_targ, rel_targ, new_trans.translation, 'offset', self.human_controller.translation_offset)
        base_action = amass_human_controller.AmassHumanController.transformAction(new_pos, new_trans)
        # print(new_trans, robot_pos)
        # breakpoint()
        kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action
        return super().step(*args, is_last_action=is_last_action, **kwargs)
