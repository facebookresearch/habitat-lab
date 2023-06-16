#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pickle

from typing import Any, Optional

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from gym import spaces

from habitat.datasets.ovmm.ovmm_dataset import (
    OVMMDatasetV0,
    OVMMEpisode,
)
from habitat.tasks.ovmm.sub_tasks.nav_to_obj_sensors import (
    OVMMNavToObjSucc,
    OVMMRotDistToGoal,
    TargetIoUCoverage,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToGoal,
    NavToPosSucc,
)
from habitat.tasks.rearrange.sub_tasks.pick_sensors import RearrangePickSuccess

from habitat.tasks.ovmm.sub_tasks.place_sensors import OVMMPlaceSuccess


@registry.register_sensor
class ObjectCategorySensor(Sensor):
    cls_uuid: str = "object_category"

    def __init__(
        self,
        sim,
        config,
        dataset: "OVMMDatasetV0",
        category_attribute="object_category",
        name_to_id_mapping="obj_category_to_obj_category_id",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        self._category_attribute = category_attribute
        self._name_to_id_mapping = name_to_id_mapping

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = max(
            getattr(self._dataset, self._name_to_id_mapping).values()
        )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OVMMEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        category_name = getattr(episode, self._category_attribute)
        return np.array(
            [getattr(self._dataset, self._name_to_id_mapping)[category_name]],
            dtype=np.int64,
        )


@registry.register_sensor
class ObjectEmbeddingSensor(Sensor):
    cls_uuid: str = "object_embedding"

    def __init__(
        self,
        sim,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._dimensionality = self._config.dimensionality
        with open(config.embeddings_file, "rb") as f:
            self._embeddings = pickle.load(f)

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(self._dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, *args, episode, **kwargs):
        category_name = episode.object_category
        return self._embeddings[category_name]


@registry.register_sensor
class GoalReceptacleSensor(ObjectCategorySensor):
    cls_uuid: str = "goal_receptacle"

    def __init__(
        self,
        sim,
        config,
        dataset: "OVMMDatasetV0",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            sim=sim,
            config=config,
            dataset=dataset,
            category_attribute="goal_recep_category",
            name_to_id_mapping="recep_category_to_recep_category_id",
        )


@registry.register_sensor
class StartReceptacleSensor(ObjectCategorySensor):
    cls_uuid: str = "start_receptacle"

    def __init__(
        self,
        sim,
        config,
        dataset: "OVMMDatasetV0",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            sim=sim,
            config=config,
            dataset=dataset,
            category_attribute="start_recep_category",
            name_to_id_mapping="recep_category_to_recep_category_id",
        )


@registry.register_sensor
class ObjectSegmentationSensor(Sensor):
    cls_uuid: str = "object_segmentation"
    panoptic_uuid: str = "robot_head_panoptic"

    def __init__(
        self,
        sim,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._blank_out_prob = self._config.blank_out_prob
        self._sim = sim
        self._instance_ids_start = self._sim.habitat_config.instance_ids_start
        self._resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(
                self._resolution[0],
                self._resolution[1],
                1,
            ),
            low=0,
            high=1,
            dtype=np.uint8,
        )

    def get_observation(self, observations, *args, episode, task, **kwargs):
        if np.random.random() < self._blank_out_prob:
            return np.zeros_like(
                observations[self.panoptic_uuid], dtype=np.uint8
            )
        else:
            segmentation_sensor = np.zeros_like(
                observations[self.panoptic_uuid], dtype=np.uint8
            )
            for g in episode.candidate_objects_hard:
                segmentation_sensor = segmentation_sensor | (
                    observations[self.panoptic_uuid]
                    == self._sim.scene_obj_ids[int(g.object_id)]
                    + self._instance_ids_start
                )
            return segmentation_sensor


@registry.register_sensor
class RecepSegmentationSensor(ObjectSegmentationSensor):
    cls_uuid: str = "recep_segmentation"

    def _get_recep_goals(self, episode):
        raise NotImplementedError

    def get_observation(self, observations, *args, episode, task, **kwargs):
        recep_goals = self._get_recep_goals(episode)
        if np.random.random() < self._config.blank_out_prob:
            return np.zeros_like(
                observations[self.panoptic_uuid], dtype=np.uint8
            )
        else:
            segmentation_sensor = np.zeros_like(
                observations[self.panoptic_uuid], dtype=np.uint8
            )
            for g in recep_goals:
                segmentation_sensor = segmentation_sensor | (
                    observations[self.panoptic_uuid]
                    == int(g.object_id)
                    + self._sim.habitat_config.instance_ids_start
                )
            return segmentation_sensor


@registry.register_sensor
class StartRecepSegmentationSensor(RecepSegmentationSensor):
    cls_uuid: str = "start_recep_segmentation"

    def _get_recep_goals(self, episode):
        return episode.candidate_start_receps


@registry.register_sensor
class GoalRecepSegmentationSensor(RecepSegmentationSensor):
    cls_uuid: str = "goal_recep_segmentation"

    def _get_recep_goals(self, episode):
        return episode.candidate_goal_receps



# Sensors for measuring success to pick goals
@registry.register_measure
class OVMMDistToPickGoal(DistToGoal):
    """Distance to the closest viewpoint of a candidate pick object"""

    cls_uuid: str = "ovmm_dist_to_pick_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMDistToPickGoal.cls_uuid

    def _get_goals(self, task, episode):
        return np.stack(
            [
                view_point.agent_state.position
                for goal in episode.candidate_objects
                for view_point in goal.view_points
            ],
            axis=0,
        )


@registry.register_measure
class OVMMRotDistToPickGoal(OVMMRotDistToGoal, Measure):
    """Angle between agent's forward vector and the vector from agent's position to the closest candidate pick object"""

    cls_uuid: str = "ovmm_rot_dist_to_pick_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMRotDistToPickGoal.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
        self._is_nav_to_obj = True


@registry.register_measure
class OVMMNavToPickSucc(NavToPosSucc):
    """Whether the agent has navigated within `success_distance` of the closest viewpoint of a candidate pick object"""

    cls_uuid: str = "ovmm_nav_to_pick_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMNavToPickSucc.cls_uuid

    @property
    def _dist_to_goal_cls_uuid(self):
        return OVMMDistToPickGoal.cls_uuid


@registry.register_measure
class OVMMNavOrientToPickSucc(OVMMNavToObjSucc):
    """Whether the agent has navigated within `success_distance` of the closest viewpoint of a candidate pick object and oriented itself within `success_angle` of the object"""

    cls_uuid: str = "ovmm_nav_orient_to_pick_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMNavOrientToPickSucc.cls_uuid

    @property
    def _nav_to_pos_succ_cls_uuid(self):
        return OVMMNavToPickSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OVMMRotDistToPickGoal.cls_uuid

    @property
    def _target_iou_coverage_cls_uuid(self):
        return PickGoalIoUCoverage.cls_uuid


@registry.register_measure
class PickGoalIoUCoverage(TargetIoUCoverage):
    """IoU coverage of the target object"""

    cls_uuid: str = "pick_goal_iou_coverage"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PickGoalIoUCoverage.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
        self._is_nav_to_obj = True


# Sensors for measuring success to place goals
@registry.register_measure
class OVMMDistToPlaceGoal(DistToGoal):
    """Distance to the closest viewpoint of a candidate place receptacle"""

    cls_uuid: str = "ovmm_dist_to_place_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMDistToPlaceGoal.cls_uuid

    def _get_goals(self, task, episode):
        return np.stack(
            [
                view_point.agent_state.position
                for goal in episode.candidate_goal_receps
                for view_point in goal.view_points
            ],
            axis=0,
        )


@registry.register_measure
class OVMMRotDistToPlaceGoal(OVMMRotDistToGoal):
    """Angle between agent's forward vector and the vector from agent's position to the center of the closest candidate place receptacle"""

    cls_uuid: str = "ovmm_rot_dist_to_place_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMRotDistToPlaceGoal.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
        self._is_nav_to_obj = False


@registry.register_measure
class OVMMNavToPlaceSucc(NavToPosSucc):
    """Whether the agent has navigated within `success_distance` of the center of the closest candidate place receptacle"""

    cls_uuid: str = "ovmm_nav_to_place_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMNavToPlaceSucc.cls_uuid

    @property
    def _dist_to_goal_cls_uuid(self):
        return OVMMDistToPlaceGoal.cls_uuid


@registry.register_measure
class PlaceGoalIoUCoverage(TargetIoUCoverage):
    """IoU coverage of the target place receptacle"""

    cls_uuid: str = "place_goal_iou_coverage"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PlaceGoalIoUCoverage.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
        self._is_nav_to_obj = False


@registry.register_measure
class OVMMNavOrientToPlaceSucc(OVMMNavToObjSucc):
    """Whether the agent has navigated within `success_distance` of the center of the closest candidate place receptacle and oriented itself within `success_angle` of the receptacle"""

    cls_uuid: str = "ovmm_nav_orient_to_place_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMNavOrientToPlaceSucc.cls_uuid

    @property
    def _nav_to_pos_succ_cls_uuid(self):
        return OVMMNavToPlaceSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OVMMRotDistToPlaceGoal.cls_uuid
        
    @property
    def _target_iou_coverage_cls_uuid(self):
        return PlaceGoalIoUCoverage.cls_uuid


@registry.register_measure
class OVMMFindObjectPhaseSuccess(Measure):
    """Whether the agent has successfully found an object"""

    cls_uuid: str = "ovmm_find_object_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMFindObjectPhaseSuccess.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )

    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OVMMNavOrientToPickSucc.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        nav_orient_to_pick_succ = task.measurements.measures[
            OVMMNavOrientToPickSucc.cls_uuid
        ].get_metric()
        self._metric = nav_orient_to_pick_succ or self._metric


@registry.register_measure
class OVMMPickObjectPhaseSuccess(Measure):
    """Whether the agent has successfully completed the pick object stage successfully"""

    cls_uuid: str = "ovmm_pick_object_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMPickObjectPhaseSuccess.cls_uuid
    
    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
    
    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OVMMFindObjectPhaseSuccess.cls_uuid, RearrangePickSuccess.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        find_object_phase_success = task.measurements.measures[
            OVMMFindObjectPhaseSuccess.cls_uuid
        ].get_metric()
        did_pick_object = task.measurements.measures[
            RearrangePickSuccess.cls_uuid
        ].get_metric()
        self._metric = (find_object_phase_success and did_pick_object) or self._metric


@registry.register_measure
class OVMMFindRecepPhaseSuccess(Measure):
    """Whether the agent has successfully completed the find receptacle stage"""

    cls_uuid: str = "ovmm_find_recep_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMFindRecepPhaseSuccess.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )

    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OVMMNavOrientToPlaceSucc.cls_uuid, OVMMPickObjectPhaseSuccess.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        pick_object_phase_success = task.measurements.measures[
            OVMMPickObjectPhaseSuccess.cls_uuid
        ].get_metric()
        nav_orient_to_place_succ = task.measurements.measures[
            OVMMNavOrientToPlaceSucc.cls_uuid
        ].get_metric()
        self._metric = (pick_object_phase_success and nav_orient_to_place_succ) or self._metric


@registry.register_measure
class OVMMPlaceObjectPhaseSuccess(Measure):
    """Whether the agent has successfully completed the place object stage"""

    cls_uuid: str = "ovmm_place_object_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMPlaceObjectPhaseSuccess.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )

    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OVMMFindRecepPhaseSuccess.cls_uuid, OVMMPlaceSuccess.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        find_recep_phase_success = task.measurements.measures[
            OVMMFindRecepPhaseSuccess.cls_uuid
        ].get_metric()
        place_success = task.measurements.measures[
            OVMMPlaceSuccess.cls_uuid
        ].get_metric()
        self._metric = (find_recep_phase_success and place_success) or self._metric

