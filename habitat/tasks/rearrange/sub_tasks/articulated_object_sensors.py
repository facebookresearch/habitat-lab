#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward


@registry.register_measure
class ArtObjState(Measure):
    cls_uuid: str = "art_obj_state"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ArtObjState.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = task.get_use_marker().get_targ_js()


@registry.register_measure
class ArtObjSuccess(Measure):
    cls_uuid: str = "art_obj_success"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ArtObjSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        dist = task.success_js_state - task.get_use_marker().get_targ_js()
        if self._config.USE_ABSOLUTE_DISTANCE:
            self._metric = abs(dist) < self._config.SUCCESS_DIST_THRESHOLD
        else:
            self._metric = dist < self._config.SUCCESS_DIST_THRESHOLD


@registry.register_measure
class ArtObjReward(RearrangeReward):
    cls_uuid: str = "art_obj_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._metric = None

        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ArtObjReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [ArtObjState.cls_uuid, ArtObjSuccess.cls_uuid]
        )
        link_state = task.measurements.measures[
            ArtObjState.cls_uuid
        ].get_metric()

        self._prev_art_state = link_state
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )
        reward = self._metric
        link_state = task.measurements.measures[
            ArtObjState.cls_uuid
        ].get_metric()

        is_succ = task.measurements.measures[
            ArtObjSuccess.cls_uuid
        ].get_metric()

        cur_dist = abs(link_state - task.success_js_state)
        prev_dist = abs(self._prev_art_state - task.success_js_state)

        dist_diff = prev_dist - cur_dist
        reward += self._config.DIST_REWARD * dist_diff
        if is_succ:
            reward += self._config.SUCCESS_REWARD

        if (
            task._sim.grasp_mgr.is_grasped
            and task._sim.grasp_mgr.snapped_marker_id != task.use_marker_name
        ):
            reward -= self._config.WRONG_GRASP_PEN
            if self._config.WRONG_GRASP_END:
                task.should_end = True

        self._metric = reward
