#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.cat_nav_to_obj_sensors import (
    OvmmRotDistToGoal,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToGoal,
    NavToObjSuccess,
    NavToPosSucc,
)
from habitat.tasks.rearrange.sub_tasks.pick_sensors import RearrangePickSuccess

from habitat.tasks.rearrange.sub_tasks.ovmm_place_sensors import OvmmPlaceSuccess



# Sensors for measuring success to pick goals
@registry.register_measure
class OvmmDistToPickGoal(DistToGoal):
    """Distance to the closest viewpoint of a candidate pick object"""

    cls_uuid: str = "ovmm_dist_to_pick_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmDistToPickGoal.cls_uuid

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
class OvmmRotDistToPickGoal(OvmmRotDistToGoal, Measure):
    """Angle between agent's forward vector and the vector from agent's position to the closest candidate pick object"""

    cls_uuid: str = "ovmm_rot_dist_to_pick_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmRotDistToPickGoal.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
        self._is_nav_to_obj = True


@registry.register_measure
class OvmmNavToPickSucc(NavToPosSucc):
    """Whether the agent has navigated within `success_distance` of the closest viewpoint of a candidate pick object"""

    cls_uuid: str = "ovmm_nav_to_pick_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmNavToPickSucc.cls_uuid

    @property
    def _dist_to_goal_cls_uuid(self):
        return OvmmDistToPickGoal.cls_uuid


@registry.register_measure
class OvmmNavOrientToPickSucc(NavToObjSuccess):
    """Whether the agent has navigated within `success_distance` of the closest viewpoint of a candidate pick object and oriented itself within `success_angle` of the object"""

    cls_uuid: str = "ovmm_nav_orient_to_pick_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmNavOrientToPickSucc.cls_uuid

    @property
    def _nav_to_pos_succ_cls_uuid(self):
        return OvmmNavToPickSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OvmmRotDistToPickGoal.cls_uuid


# Sensors for measuring success to place goals
@registry.register_measure
class OvmmDistToPlaceGoal(DistToGoal):
    """Distance to the closest viewpoint of a candidate place receptacle"""

    cls_uuid: str = "ovmm_dist_to_place_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmDistToPlaceGoal.cls_uuid

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
class OvmmRotDistToPlaceGoal(OvmmRotDistToGoal):
    """Angle between agent's forward vector and the vector from agent's position to the center of the closest candidate place receptacle"""

    cls_uuid: str = "ovmm_rot_dist_to_place_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmRotDistToPlaceGoal.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
        self._is_nav_to_obj = False


@registry.register_measure
class OvmmNavToPlaceSucc(NavToPosSucc):
    """Whether the agent has navigated within `success_distance` of the center of the closest candidate place receptacle"""

    cls_uuid: str = "ovmm_nav_to_place_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmNavToPlaceSucc.cls_uuid

    @property
    def _dist_to_goal_cls_uuid(self):
        return OvmmDistToPlaceGoal.cls_uuid


@registry.register_measure
class OvmmNavOrientToPlaceSucc(NavToObjSuccess):
    """Whether the agent has navigated within `success_distance` of the center of the closest candidate place receptacle and oriented itself within `success_angle` of the receptacle"""

    cls_uuid: str = "ovmm_nav_orient_to_place_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmNavOrientToPlaceSucc.cls_uuid

    @property
    def _nav_to_pos_succ_cls_uuid(self):
        return OvmmNavToPlaceSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OvmmRotDistToPlaceGoal.cls_uuid


@registry.register_measure
class OvmmFindObjectPhaseSuccess(Measure):
    """Whether the agent has successfully found an object"""

    cls_uuid: str = "ovmm_find_object_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmFindObjectPhaseSuccess.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )

    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OvmmNavOrientToPickSucc.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        nav_orient_to_pick_succ = task.measurements.measures[
            OvmmNavOrientToPickSucc.cls_uuid
        ].get_metric()
        self._metric = nav_orient_to_pick_succ or self._metric


@registry.register_measure
class OvmmPickObjectPhaseSuccess(Measure):
    """Whether the agent has successfully completed the pick object stage successfully"""

    cls_uuid: str = "ovmm_pick_object_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmPickObjectPhaseSuccess.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )

    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OvmmFindObjectPhaseSuccess.cls_uuid, RearrangePickSuccess.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        find_object_phase_success = task.measurements.measures[
            OvmmFindObjectPhaseSuccess.cls_uuid
        ].get_metric()
        did_pick_object = task.measurements.measures[
            RearrangePickSuccess.cls_uuid
        ].get_metric()
        self._metric = (find_object_phase_success and did_pick_object) or self._metric


@registry.register_measure
class OvmmFindRecepPhaseSuccess(Measure):
    """Whether the agent has successfully completed the find receptacle stage"""

    cls_uuid: str = "ovmm_find_recep_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmFindRecepPhaseSuccess.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )

    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OvmmNavOrientToPlaceSucc.cls_uuid, OvmmPickObjectPhaseSuccess.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        pick_object_phase_success = task.measurements.measures[
            OvmmPickObjectPhaseSuccess.cls_uuid
        ].get_metric()
        nav_orient_to_place_succ = task.measurements.measures[
            OvmmNavOrientToPlaceSucc.cls_uuid
        ].get_metric()
        self._metric = (pick_object_phase_success and nav_orient_to_place_succ) or self._metric


@registry.register_measure
class OvmmPlaceObjectPhaseSuccess(Measure):
    """Whether the agent has successfully completed the place object stage"""

    cls_uuid: str = "ovmm_place_object_phase_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmPlaceObjectPhaseSuccess.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )

    def reset_metric(self, *args, episode, task,  **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [OvmmFindRecepPhaseSuccess.cls_uuid, OvmmPlaceSuccess.cls_uuid]
        )
        self._metric = False


    def update_metric(self, episode, task, *args, **kwargs):
        find_recep_phase_success = task.measurements.measures[
            OvmmFindRecepPhaseSuccess.cls_uuid
        ].get_metric()
        place_success = task.measurements.measures[
            OvmmPlaceSuccess.cls_uuid
        ].get_metric()
        self._metric = (find_recep_phase_success and place_success) or self._metric
