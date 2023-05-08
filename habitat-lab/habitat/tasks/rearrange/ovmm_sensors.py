#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import NavToPosSucc, NavToObjSuccess
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry


from habitat.tasks.rearrange.sub_tasks.pick_sensors import (
    DidPickObjectMeasure,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToGoal
)
from habitat.tasks.rearrange.sub_tasks.cat_nav_to_obj_sensors import (
    CatNavRotDistToGoal
)
import numpy as np

# Sensors for measuring success to pick goals
@registry.register_measure
class DistToPickGoal(DistToGoal):
    cls_uuid: str = "dist_to_pick_goal"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DistToPickGoal.cls_uuid
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
class RotDistToPickGoal(CatNavRotDistToGoal, Measure):
    cls_uuid: str = "rot_dist_to_pick_goal"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RotDistToPickGoal.cls_uuid
    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(*args, sim=sim, config=config, dataset=dataset, task=task, **kwargs)
        self._is_nav_to_obj = True


@registry.register_measure
class NavToPickSucc(NavToPosSucc):
    cls_uuid: str = "nav_to_pick_succ"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToPickSucc.cls_uuid
    @property
    def _dist_to_goal_cls_uuid(self):
        return DistToPickGoal.cls_uuid


@registry.register_measure
class NavOrientToPickSucc(NavToObjSuccess):
    cls_uuid: str = "nav_orient_to_pick_succ"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavOrientToPickSucc.cls_uuid
    @property
    def _nav_to_pos_succ_cls_uuid(self):
        return NavToPickSucc.cls_uuid
    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return RotDistToPickGoal.cls_uuid

# Sensors for measuring success to place goals
@registry.register_measure
class DistToPlaceGoal(DistToGoal):
    cls_uuid: str = "dist_to_place_goal"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DistToPlaceGoal.cls_uuid
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
class RotDistToPlaceGoal(CatNavRotDistToGoal):
    cls_uuid: str = "rot_dist_to_place_goal"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RotDistToPlaceGoal.cls_uuid
    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(*args, sim=sim, config=config, dataset=dataset, task=task, **kwargs)
        self._is_nav_to_obj = False


@registry.register_measure
class NavToPlaceSucc(NavToPosSucc):
    cls_uuid: str = "nav_to_place_succ"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToPlaceSucc.cls_uuid
    @property
    def _dist_to_goal_cls_uuid(self):
        return DistToPlaceGoal.cls_uuid


@registry.register_measure
class NavOrientToPlaceSucc(NavToObjSuccess):
    cls_uuid: str = "nav_orient_to_place_succ"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavOrientToPlaceSucc.cls_uuid
    @property
    def _nav_to_pos_succ_cls_uuid(self):
        return NavToPlaceSucc.cls_uuid
    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return RotDistToPlaceGoal.cls_uuid


@registry.register_measure
class PickNavToPlaceSucc(Measure):
    cls_uuid: str = "pick_nav_to_place_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PickNavToPlaceSucc.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [NavToPlaceSucc.cls_uuid, DidPickObjectMeasure.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def __init__(self, *args, config, **kwargs):
        self._config = config
        super().__init__(*args, config=config, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        nav_to_pos_success = task.measurements.measures[
            NavToPlaceSucc.cls_uuid
        ].get_metric()

        did_pick_object = task.measurements.measures[
            DidPickObjectMeasure.cls_uuid
        ].get_metric()
        self._metric = nav_to_pos_success and did_pick_object


@registry.register_measure
class PickNavOrientToPlaceSucc(Measure):
    cls_uuid: str = "pick_nav_orient_to_place_succ"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PickNavOrientToPlaceSucc.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [NavOrientToPlaceSucc.cls_uuid, DidPickObjectMeasure.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def __init__(self, *args, config, **kwargs):
        self._config = config
        super().__init__(*args, config=config, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        nav_to_obj_success = task.measurements.measures[
            NavOrientToPlaceSucc.cls_uuid
        ].get_metric()

        did_pick_object = task.measurements.measures[
            DidPickObjectMeasure.cls_uuid
        ].get_metric()
        self._metric = nav_to_obj_success and did_pick_object

