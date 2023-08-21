#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
from typing import Optional, cast

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import NavToInfo
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_task(name="RearrangeCompositeTask-v0")
class CompositeTask(RearrangeTask):
    """
    All tasks using a combination of sub-tasks (skills) should utilize this task.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        task_spec_path = osp.join(
            config.task_spec_base_path,
            config.task_spec + ".yaml",
        )

        self.pddl_problem = PddlProblem(
            config.pddl_domain_def,
            task_spec_path,
            config,
        )

        super().__init__(config=config, *args, dataset=dataset, **kwargs)

        self._cur_node_idx: int = -1

    def jump_to_node(
        self, node_idx: int, episode: Episode, is_full_task: bool = False
    ) -> None:
        """
        Sequentially applies all solution actions before `node_idx`. But NOT
        including the solution action at index `node_idx`.

        :param node_idx: An integer in [0, len(solution)).
        :param is_full_task: If true, then calling reset will always the task to this solution node.
        """

        rearrange_logger.debug(
            "Jumping to node {node_idx}, is_full_task={is_full_task}"
        )
        # We don't want to reset to this node if we are in full task mode.
        if not is_full_task:
            self._cur_node_idx = node_idx

        for i in range(node_idx):
            self.pddl_problem.apply_action(self.pddl_problem.solution[i])

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        self.pddl_problem.bind_to_instance(
            self._sim, cast(RearrangeDatasetV0, self._dataset), self, episode
        )

        if self._cur_node_idx >= 0:
            self.jump_to_node(self._cur_node_idx, episode)

        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)


@registry.register_task(name="RearrangeCompositeTaskNavGoal-v0")
class CompositeTaskNavGoal(CompositeTask):
    """
    All tasks that mimic the compsite task but with navigation target
    """

    _nav_to_info: Optional[NavToInfo]

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)

        self.force_obj_to_idx = None
        self.force_recep_to_name = None
        self._object_in_hand_sample_prob = config.object_in_hand_sample_prob
        self._min_start_distance = config.min_start_distance
        # Robot will be the first agent
        self.agent_id = 0

    def _generate_snap_to_obj(self) -> int:
        # Snap the target object to the articulated_agent hand.
        target_idxs, _ = self._sim.get_targets()
        return self._sim.scene_obj_ids[target_idxs[0]]

    def _generate_nav_start_goal(self, episode, force_idx=None) -> NavToInfo:
        """
        Returns the starting information for a navigate to object task.
        """

        start_hold_obj_idx: Optional[int] = None

        # Only change the scene if this skill is not running as a sub-task
        if (
            force_idx is None
            and random.random() < self._object_in_hand_sample_prob
        ):
            start_hold_obj_idx = self._generate_snap_to_obj()

        if start_hold_obj_idx is None:
            # Select an object at random and navigate to that object.
            all_pos = self._sim.get_target_objs_start()
            if force_idx is None:
                nav_to_pos = all_pos[np.random.randint(0, len(all_pos))]
            else:
                nav_to_pos = all_pos[force_idx]
        else:
            # Select a goal at random and navigate to that goal.
            _, all_pos = self._sim.get_targets()
            nav_to_pos = all_pos[np.random.randint(0, len(all_pos))]

        def filter_func(start_pos, _):
            return (
                np.linalg.norm(start_pos - nav_to_pos)
                > self._min_start_distance
            )

        (
            articulated_agent_pos,
            articulated_agent_angle,
        ) = self._sim.set_articulated_agent_base_to_random_point(
            filter_func=filter_func
        )
        return NavToInfo(
            nav_goal_pos=nav_to_pos,
            articulated_agent_start_pos=articulated_agent_pos,
            articulated_agent_start_angle=articulated_agent_angle,
            start_hold_obj_idx=start_hold_obj_idx,
        )

    @property
    def nav_goal_pos(self):
        return self._nav_to_info.nav_goal_pos

    def reset(self, episode: Episode):
        # Process the nav position
        self._nav_to_info = self._generate_nav_start_goal(
            episode, force_idx=self.force_obj_to_idx
        )

        self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_pos = (
            self._nav_to_info.articulated_agent_start_pos
        )
        self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_rot = (
            self._nav_to_info.articulated_agent_start_angle
        )

        # Reset as usual
        super().reset(episode)
        self.pddl_problem.bind_to_instance(
            self._sim, cast(RearrangeDatasetV0, self._dataset), self, episode
        )
        if self._cur_node_idx >= 0:
            self.jump_to_node(self._cur_node_idx, episode)

        # Snap the object
        if self._nav_to_info.start_hold_obj_idx is not None:
            if self._sim.get_agent_data(self.agent_id).grasp_mgr.is_grasped:
                raise ValueError(
                    f"Attempting to grasp {self._nav_to_info.start_hold_obj_idx} even though object is already grasped"
                )
            rearrange_logger.debug(
                f"Forcing to grasp object {self._nav_to_info.start_hold_obj_idx}"
            )
            self._sim.get_agent_data(self.agent_id).grasp_mgr.snap_to_obj(
                self._nav_to_info.start_hold_obj_idx, force=True
            )

        if self._sim.habitat_config.debug_render:
            # Visualize the position the agent is navigating to.
            self._sim.viz_ids["nav_targ_pos"] = self._sim.visualize_position(
                self._nav_to_info.nav_goal_pos,
                self._sim.viz_ids["nav_targ_pos"],
                r=0.2,
            )

        self._sim.maybe_update_articulated_agent()

        return self._get_observations(episode)
