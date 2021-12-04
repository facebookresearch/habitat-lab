#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward


@registry.register_sensor
class MarkerRelPosSensor(Sensor):
    cls_uuid: str = "marker_rel_pos"

    def __init__(self, sim, config, *args, task, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MarkerRelPosSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        marker = self._sim.get_marker(self._task.use_marker_name)
        ee_trans = self._sim.robot.ee_transform
        rel_marker_pos = ee_trans.inverted().transform_point(
            marker.get_current_position()
        )

        return np.array(rel_marker_pos)


@registry.register_sensor
class ArtJointSensor(Sensor):
    cls_uuid: str = "marker_js"

    def __init__(self, sim, config, *args, task, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task

    def _get_uuid(self, *args, **kwargs):
        return ArtJointSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(2,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        js = self._task.get_use_marker().get_targ_js()
        js_vel = self._task.get_use_marker().get_targ_js_vel()
        return np.array([js, js_vel]).reshape((2,))


@registry.register_sensor
class ArtJointSensorNoVel(Sensor):
    cls_uuid: str = "marker_js_no_vel"

    def __init__(self, sim, config, *args, task, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task

    def _get_uuid(self, *args, **kwargs):
        return ArtJointSensorNoVel.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        js = self._task.get_use_marker().get_targ_js()
        return np.array([js]).reshape((1,))


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
        # If not absolute distance, we can have a joint state greater than the
        # target.
        if self._config.USE_ABSOLUTE_DISTANCE:
            self._metric = abs(dist) < self._config.SUCCESS_DIST_THRESHOLD
        else:
            self._metric = dist < self._config.SUCCESS_DIST_THRESHOLD


@registry.register_measure
class EndEffectorDistToMarker(Measure):
    cls_uuid: str = "ee_dist_to_marker"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorDistToMarker.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = np.linalg.norm(
            observations[MarkerRelPosSensor.cls_uuid]
        )


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

        dist_to_marker = task.measurements.measures[
            EndEffectorDistToMarker.cls_uuid
        ].get_metric()

        self._prev_art_state = link_state
        self._prev_has_grasped = task._sim.grasp_mgr.is_grasped
        self._prev_ee_dist_to_marker = dist_to_marker
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

        cur_dist = abs(link_state - task.success_js_state)
        prev_dist = abs(self._prev_art_state - task.success_js_state)

        dist_diff = prev_dist - cur_dist
        reward += self._config.DIST_REWARD * dist_diff

        cur_has_grasped = task._sim.grasp_mgr.is_grasped

        cur_ee_dist_to_marker = task.measurements.measures[
            EndEffectorDistToMarker.cls_uuid
        ].get_metric()
        if not cur_has_grasped:
            dist_diff = self._prev_ee_dist_to_marker - cur_ee_dist_to_marker
            reward += self._config.MARKER_DIST_REWARD * dist_diff

        if cur_has_grasped and not self._prev_has_grasped:
            if task._sim.grasp_mgr.snapped_marker_id != task.use_marker_name:
                # Grasped wrong object
                reward -= self._config.WRONG_GRASP_PEN
                if self._config.WRONG_GRASP_END:
                    task.should_end = True
            else:
                # Grasped right object
                reward += self._config.GRASP_REWARD

        self._prev_ee_dist_to_marker = cur_ee_dist_to_marker
        self._prev_art_state = link_state
        self._prev_has_grasped = cur_has_grasped
        self._metric = reward
