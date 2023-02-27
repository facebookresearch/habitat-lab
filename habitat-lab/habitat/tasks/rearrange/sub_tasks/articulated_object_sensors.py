#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.rearrange_sensors import (
    DoesWantTerminate,
    EndEffectorToRestDistance,
    RearrangeReward,
)
from habitat.tasks.rearrange.utils import (
    UsesArticulatedAgentInterface,
    rearrange_logger,
)


@registry.register_sensor
class MarkerRelPosSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Tracks the relative position of a marker to the robot end-effector
    specified by `use_marker_name` in the task. This `use_marker_name` must
    exist in the task and refer to the name of a marker in the simulator.
    """

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
        marker = self._task.get_use_marker()
        ee_trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        rel_marker_pos = ee_trans.inverted().transform_point(
            marker.get_current_position()
        )

        return np.array(rel_marker_pos)


@registry.register_sensor
class ArtJointSensor(Sensor):
    """
    Gets the joint state (position and velocity) of the articulated object
    specified by the `use_marker_name` property in the task object.
    """

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
        return np.array([js, js_vel], dtype=np.float32).reshape((2,))


@registry.register_sensor
class ArtJointSensorNoVel(Sensor):
    """
    Gets the joint state (just position) of the articulated object
    specified by the `use_marker_name` property in the task object.
    """

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
    """
    Measures the current joint state of the target articulated object.
    """

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
class ArtObjAtDesiredState(Measure):
    cls_uuid: str = "art_obj_at_desired_state"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ArtObjAtDesiredState.cls_uuid

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
        if self._config.use_absolute_distance:
            self._metric = abs(dist) < self._config.success_dist_threshold
        else:
            self._metric = dist < self._config.success_dist_threshold


@registry.register_measure
class ArtObjSuccess(Measure):
    """
    Measures if the target articulated object joint state is at the success criteria.
    """

    cls_uuid: str = "art_obj_success"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._sim = sim
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
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()
        is_art_obj_state_succ = task.measurements.measures[
            ArtObjAtDesiredState.cls_uuid
        ].get_metric()

        called_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()

        # If not absolute distance, we can have a joint state greater than the
        # target.
        self._metric = (
            is_art_obj_state_succ
            and ee_to_rest_distance < self._config.rest_dist_threshold
            and not self._sim.grasp_mgr.is_grasped
        )
        if self._config.must_call_stop:
            if called_stop:
                task.should_end = True
            else:
                self._metric = False


@registry.register_measure
class EndEffectorDistToMarker(UsesArticulatedAgentInterface, Measure):
    """
    The distance of the end-effector to the target marker on the articulated object.
    """

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

    def update_metric(self, *args, task, **kwargs):
        marker = task.get_use_marker()
        ee_trans = task._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        rel_marker_pos = ee_trans.inverted().transform_point(
            marker.get_current_position()
        )

        self._metric = np.linalg.norm(rel_marker_pos)


@registry.register_measure
class ArtObjReward(RearrangeReward):
    """
    A general reward definition for any tasks involving manipulating articulated objects.
    """

    cls_uuid: str = "art_obj_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._metric = None

        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ArtObjReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ArtObjState.cls_uuid,
                ArtObjSuccess.cls_uuid,
                EndEffectorToRestDistance.cls_uuid,
                ArtObjAtDesiredState.cls_uuid,
            ],
        )
        link_state = task.measurements.measures[
            ArtObjState.cls_uuid
        ].get_metric()

        dist_to_marker = task.measurements.measures[
            EndEffectorDistToMarker.cls_uuid
        ].get_metric()

        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        self._prev_art_state = link_state
        self._any_has_grasped = task._sim.grasp_mgr.is_grasped
        self._prev_ee_dist_to_marker = dist_to_marker
        self._prev_ee_to_rest = ee_to_rest_distance
        self._any_at_desired_state = False
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

        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        is_art_obj_state_succ = task.measurements.measures[
            ArtObjAtDesiredState.cls_uuid
        ].get_metric()

        cur_dist = abs(link_state - task.success_js_state)
        prev_dist = abs(self._prev_art_state - task.success_js_state)

        # Dense reward to the target articulated object state.
        dist_diff = prev_dist - cur_dist
        if not is_art_obj_state_succ:
            reward += self._config.art_dist_reward * dist_diff

        cur_has_grasped = task._sim.grasp_mgr.is_grasped

        cur_ee_dist_to_marker = task.measurements.measures[
            EndEffectorDistToMarker.cls_uuid
        ].get_metric()
        if cur_has_grasped and not self._any_has_grasped:
            if task._sim.grasp_mgr.snapped_marker_id != task.use_marker_name:
                # Grasped wrong marker
                reward -= self._config.wrong_grasp_pen
                if self._config.wrong_grasp_end:
                    rearrange_logger.debug(
                        "Grasped wrong marker, ending episode."
                    )
                    task.should_end = True
            else:
                # Grasped right marker
                reward += self._config.grasp_reward
            self._any_has_grasped = True

        if is_art_obj_state_succ:
            if not self._any_at_desired_state:
                reward += self._config.art_at_desired_state_reward
                self._any_at_desired_state = True
            # Give the reward based on distance to the resting position.
            ee_dist_change = self._prev_ee_to_rest - ee_to_rest_distance
            reward += self._config.ee_dist_reward * ee_dist_change
        elif not cur_has_grasped:
            # Give the reward based on distance to the handle
            dist_diff = self._prev_ee_dist_to_marker - cur_ee_dist_to_marker
            reward += self._config.marker_dist_reward * dist_diff

        self._prev_ee_to_rest = ee_to_rest_distance

        self._prev_ee_dist_to_marker = cur_ee_dist_to_marker
        self._prev_art_state = link_state
        self._metric = reward
