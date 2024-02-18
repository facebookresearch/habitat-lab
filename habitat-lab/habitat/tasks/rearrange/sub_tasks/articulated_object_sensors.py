#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import magnum as mn
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
    get_camera_object_angle,
    get_camera_transform,
    rearrange_logger,
)
from habitat.utils.geometry_utils import (
    cam_pose_from_opengl_to_opencv,
    cam_pose_from_xzy_to_xyz,
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
class HandleBBoxSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Detect handle and return bbox
    """

    cls_uuid: str = "handle_bbox"

    def __init__(self, sim, config, *args, task, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task
        self._height = config.height
        self._width = config.width
        self._bbox_sixe = config.bbox_sixe

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HandleBBoxSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(
                config.height,
                config.width,
                1,
            ),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_image_coordinate_from_world_coordinate(
        self, point: np.ndarray, target_key: str
    ) -> Tuple[int, int, bool]:
        """Project point in the world frame to the image plane"""
        # Get the camera info
        fs_w, fs_h, cam_pose = self.get_camera_param(target_key)

        # Do projection
        point_cam_coord = np.linalg.inv(cam_pose) @ [
            point[0],
            point[1],
            point[2],
            1,
        ]

        # For image width coordinate
        w = self._width / 2.0 + (
            fs_w * point_cam_coord[0] / point_cam_coord[-2]
        )
        # For image height coordinate
        h = self._height / 2.0 + (
            fs_h * point_cam_coord[1] / point_cam_coord[-2]
        )

        # check if the point is in front of the camera
        is_in_front_of_camera = point_cam_coord[-2] > 0
        return (w, h, is_in_front_of_camera)

    def get_camera_param(
        self, target_key: str
    ) -> Tuple[float, float, np.ndarray]:
        """Get the camera parameters from the agent's sensor"""
        agent_id = 0 if self.agent_id is None else self.agent_id

        # Get focal length
        fov = (
            float(self._sim.agents[agent_id]._sensors[target_key].hfov)
            * np.pi
            / 180
        )
        fs_w = self._width / (2 * np.tan(fov / 2.0))
        fs_h = self._height / (2 * np.tan(fov / 2.0))

        # Get the camera pose
        hab_cam_T = (
            self._sim.agents[agent_id]
            ._sensors["articulated_agent_arm_rgb"]
            .render_camera.camera_matrix.inverted()
        )
        world_T_cam = cam_pose_from_xzy_to_xyz(
            cam_pose_from_opengl_to_opencv(np.array(hab_cam_T))
        )
        return fs_w, fs_h, world_T_cam

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Get a correct observation space
        if self.agent_id is None:
            target_key = "articulated_agent_arm_rgb"
            assert target_key in observations
        else:
            target_key = f"agent_{self.agent_id}_articulated_agent_arm_rgb"
            assert target_key in observations

        # Set the image size
        img = np.zeros((self._height, self._width, 1))

        # Get the handle location
        handle_pos = self._task.get_use_marker().current_transform.translation
        # We correct the coordinate from openGL to openCV
        handle_pos = np.array([handle_pos[0], -handle_pos[2], handle_pos[1]])

        # Get the pixel coordinate in 2D
        (
            w,
            h,
            is_in_front_of_camera,
        ) = self.get_image_coordinate_from_world_coordinate(
            handle_pos, target_key
        )

        if is_in_front_of_camera:
            # Clip the width and length
            w_low = int(np.clip(w - self._bbox_sixe, 0, self._width))
            w_high = int(np.clip(w + self._bbox_sixe, 0, self._width))
            h_low = int(np.clip(h - self._bbox_sixe, 0, self._height))
            h_high = int(np.clip(h + self._bbox_sixe, 0, self._height))
            img[h_low:h_high, w_low:w_high, 0] = 1.0

        return np.float32(img)


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
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = task.get_use_marker().get_targ_js()


@registry.register_measure
class ArtObjAtDesiredState(Measure):
    cls_uuid: str = "art_obj_at_desired_state"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._gaze_method = config.gaze_method
        if config.center_cone_vector is None:
            self._center_cone_vector = None
        else:
            self._center_cone_vector = mn.Vector3(
                config.center_cone_vector
            ).normalized()
        if config.gaze_distance_range is None:
            self._min_dist, self._max_dist = 0, 0
        else:
            self._min_dist, self._max_dist = config.gaze_distance_range
        self._center_cone_angle_threshold = np.deg2rad(
            config.center_cone_angle_threshold
        )
        self._sim = sim
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
            **kwargs,
        )

    def _get_camera_object_angle(self, obj_pos):
        """Calculates angle between gripper line-of-sight and given global position."""
        # Get the camera transformation
        cam_T = get_camera_transform(self._sim.articulated_agent)

        # Get angle between (normalized) location and the vector that the camera should
        # look at
        obj_angle = get_camera_object_angle(
            cam_T, obj_pos, self._center_cone_vector
        )
        return obj_angle

    def update_metric(self, *args, episode, task, observations, **kwargs):
        # Check if the robot gazes the target
        if self._gaze_method:
            # Get distance
            handle_pos = task.get_use_marker().get_current_position()
            ee_pos = self._sim.articulated_agent.ee_transform().translation
            dist = np.linalg.norm(handle_pos - ee_pos)
            # Get gaze angle
            obj_angle = self._get_camera_object_angle(handle_pos)
            # Return metric
            if (
                dist > self._min_dist
                and dist < self._max_dist
                and obj_angle < self._center_cone_angle_threshold
            ):
                self._metric = True
            else:
                self._metric = False
        else:
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
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()
        is_art_obj_state_succ = task.measurements.measures[
            ArtObjAtDesiredState.cls_uuid
        ].get_metric()

        if self._config.must_call_stop:
            called_stop = task.measurements.measures[
                DoesWantTerminate.cls_uuid
            ].get_metric()

        # If not absolute distance, we can have a joint state greater than the
        # target.
        self._metric = (
            is_art_obj_state_succ
            and (
                ee_to_rest_distance < self._config.rest_dist_threshold
                or self._config.rest_dist_threshold == -1
            )
            and (
                not self._sim.grasp_mgr.is_grasped or self._config.gaze_method
            )
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
            **kwargs,
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
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
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

        if not self._config.gaze_method:
            cur_dist = 0
            prev_dist = 0
        else:
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
            if not self._config.gaze_method:
                if not self._any_at_desired_state:
                    reward += self._config.art_at_desired_state_reward
                    self._any_at_desired_state = True
                # Give the reward based on distance to the resting position.
                ee_dist_change = self._prev_ee_to_rest - ee_to_rest_distance
                reward += self._config.ee_dist_reward * ee_dist_change
        elif not cur_has_grasped or self._config.gaze_method:
            # Give the reward based on distance to the handle
            dist_diff = self._prev_ee_dist_to_marker - cur_ee_dist_to_marker
            reward += self._config.marker_dist_reward * dist_diff

        self._prev_ee_to_rest = ee_to_rest_distance

        self._prev_ee_dist_to_marker = cur_ee_dist_to_marker
        self._prev_art_state = link_state
        self._metric = reward
