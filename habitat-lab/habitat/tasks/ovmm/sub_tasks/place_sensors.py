import numpy as np

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.place_sensors import (
    PlacementStability,
    PlaceReward,
    PlaceSuccess,
)
from habitat.tasks.rearrange.utils import UsesRobotInterface


@registry.register_measure
class OVMMObjectToPlaceGoalDistance(Measure):
    """
    Euclidean distance from the target object to the goal.
    """

    cls_uuid: str = "ovmm_object_to_place_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMObjectToPlaceGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, **kwargs):
        # compute distance from currently picked object to candidate_goal_receps
        picked_idx = task._picked_object_idx
        goal_pos = np.array(
            [g.position for g in episode.candidate_goal_receps]
        )
        scene_pos = self._sim.get_scene_pos()
        object_pos = scene_pos[picked_idx]
        distances = np.linalg.norm(object_pos - goal_pos, ord=2, axis=-1)
        # distance to the closest goal
        self._metric = {str(picked_idx): np.min(distances)}


@registry.register_measure
class OVMMEEToPlaceGoalDistance(UsesRobotInterface, Measure):
    """
    Euclidean distance from the end-effector to the goal.
    """

    cls_uuid: str = "ovmm_ee_to_place_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMEEToPlaceGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, **kwargs):
        # compute distance from end effector to candidate_goal_receps
        picked_idx = task._picked_object_idx
        ee_pos = self._sim.get_robot_data(
            self.robot_id
        ).robot.ee_transform.translation
        goal_pos = np.array(
            [g.position for g in episode.candidate_goal_receps]
        )
        distances = np.linalg.norm(ee_pos - goal_pos, ord=2, axis=-1)
        # distance to the closest goal
        self._metric = {str(picked_idx): np.min(distances)}


@registry.register_measure
class ObjAnywhereOnGoal(Measure):
    cls_uuid: str = "obj_anywhere_on_goal"

    def __init__(self, sim, config, *args, **kwargs):
        self._config = config
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAnywhereOnGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._sim.perform_discrete_collision_detection()
        cps = self._sim.get_physics_contact_points()
        MAX_FLOOR_HEIGHT = 0.05
        picked_idx = task._picked_object_idx
        self._metric = {str(picked_idx): False}
        abs_obj_id = self._sim.scene_obj_ids[picked_idx]
        for cp in cps:
            if cp.object_id_a == abs_obj_id or cp.object_id_b == abs_obj_id:
                if cp.contact_distance < -0.01:
                    self._metric = {str(picked_idx): False}
                else:
                    other_obj_id = cp.object_id_a + cp.object_id_b - abs_obj_id
                    # Get the contact point on the other object
                    contact_point = (
                        cp.position_on_a_in_ws
                        if other_obj_id == cp.object_id_a
                        else cp.position_on_b_in_ws
                    )
                    # Check if the other object has an id that is acceptable
                    self._metric = {
                        str(picked_idx): other_obj_id
                        in self._sim.valid_goal_rec_obj_ids
                        and contact_point[1]
                        >= MAX_FLOOR_HEIGHT  # ensure that the object is not on the floor
                    }
                    # Additional check for receptacles that are not on a separate object
                    if self._metric[str(picked_idx)] and other_obj_id == -1:

                        for n, r in self._sim.receptacles.items():
                            if r.check_if_point_on_surface(
                                self._sim, contact_point
                            ):
                                self._metric = {
                                    str(picked_idx): n
                                    in self._sim.valid_goal_rec_names
                                }
                                break
                    if self._metric[str(picked_idx)]:
                        return


@registry.register_measure
class OVMMPlaceReward(PlaceReward):
    cls_uuid: str = "ovmm_place_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMPlaceReward.cls_uuid

    @property
    def _ee_to_goal_dist_cls_uuid(self):
        return OVMMEEToPlaceGoalDistance.cls_uuid

    @property
    def _obj_to_goal_dist_cls_uuid(self):
        return OVMMObjectToPlaceGoalDistance.cls_uuid

    @property
    def _obj_on_goal_cls_uuid(self):
        return ObjAnywhereOnGoal.cls_uuid


@registry.register_measure
class OVMMPlacementStability(PlacementStability):
    cls_uuid: str = "ovmm_placement_stability"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMPlacementStability.cls_uuid

    @property
    def _obj_on_goal_cls_uuid(self):
        return ObjAnywhereOnGoal.cls_uuid


@registry.register_measure
class OVMMPlaceSuccess(PlaceSuccess):
    cls_uuid: str = "ovmm_place_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMPlaceSuccess.cls_uuid

    @property
    def _placement_stability_cls_uuid(self):
        return OVMMPlacementStability.cls_uuid

    @property
    def _obj_on_goal_cls_uuid(self):
        return ObjAnywhereOnGoal.cls_uuid
