import magnum as mn
import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSimDepthSensor,
    HabitatSimRGBSensor,
)
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.utils import get_angle

# TODO: @maksymets should be accessed through Robot API
EE_GRIPPER_OFFSET = mn.Vector3(0.08, 0, 0)


@registry.register_sensor
class HeadRgbSensor(HabitatSimRGBSensor):
    def _get_uuid(self, *args, **kwargs):
        return "robot_head_rgb"


@registry.register_sensor
class HeadDepthSensor(HabitatSimDepthSensor):
    def _get_uuid(self, *args, **kwargs):
        return "robot_head_depth"


@registry.register_sensor
class ArmRgbSensor(HabitatSimRGBSensor):
    def _get_uuid(self, *args, **kwargs):
        return "arm_rgb"


@registry.register_sensor
class ArmDepthSensor(HabitatSimDepthSensor):
    def _get_uuid(self, *args, **kwargs):
        return "arm_depth"


@registry.register_sensor
class ThirdRgbSensor(HabitatSimRGBSensor):
    def _get_uuid(self, *args, **kwargs):
        return "3rd_rgb"


@registry.register_sensor
class ThirdDepthSensor(HabitatSimDepthSensor):
    def _get_uuid(self, *args, **kwargs):
        return "3rd_depth"


@registry.register_sensor
class TargetPointGoalGPSAndCompassSensor(PointGoalSensor):
    cls_uuid: str = "target_point_goal_gps_and_compass_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        target_position = self._sim.get_target_objs_start()[0]
        return self._compute_pointgoal(
            agent_position, rotation_world_agent, target_position
        )


class MultiObjSensor(PointGoalSensor):
    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._sim.get_n_targets()
        return spaces.Box(
            shape=(n_targets, 3),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_sensor
class TargetObjectSensor(MultiObjSensor):
    """
    Relative to the end effector. This is the ground truth position, accurate
    at every time step.
    """

    cls_uuid: str = "obj_cur_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        pos = scene_pos[idxs]

        ee_pos = self._sim.get_end_effector_pos()
        to_targ = pos - ee_pos
        trans = self._sim.get_robot_transform()
        for i in range(to_targ.shape[0]):
            to_targ[i] = trans.inverted().transform_vector(to_targ[i])
        return to_targ


@registry.register_sensor
class AbsTargetObjectSensor(MultiObjSensor):
    """
    This is the ground truth position
    """

    cls_uuid: str = "abs_obj_cur_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        pos = scene_pos[idxs]

        return pos


@registry.register_sensor
class TargetStartSensor(MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_start_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.robot.get_end_effector_transform()
        T_inv = global_T.inverted()
        pos = self._sim.get_target_objs_start()
        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])

        return pos

        # pos = self._sim.get_target_objs_start()
        # ee_pos = self._sim.get_end_effector_pos()
        # to_targ = pos - ee_pos
        # trans = self._sim.get_robot_transform()
        # for i in range(to_targ.shape[0]):
        #    to_targ[i] = trans.inverted().transform_vector(to_targ[i])
        # return to_targ


@registry.register_sensor
class AbsTargetStartSensor(MultiObjSensor):
    cls_uuid: str = "abs_obj_start_sensor"

    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._sim.get_n_targets()
        return spaces.Box(
            shape=(n_targets, 3),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        pos = self._sim.get_target_objs_start()
        return pos


@registry.register_sensor
class DynObjPosStartOrGoal(MultiObjSensor):
    """
    Returns the goal position if the robot is holding an object and the object
    starting position if the robot is not holding the object.
    """

    cls_uuid: str = "dyn_obj_start_or_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.robot.get_end_effector_transform()
        T_inv = global_T.inverted()

        if self._sim.snapped_obj_id is not None:
            _, pos = self._sim.get_targets()
        else:
            pos = self._sim.get_target_objs_start()
        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])

        return pos


@registry.register_sensor
class GoalSensor(MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.robot.get_end_effector_transform()
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])
        return pos


@registry.register_sensor
class AbsGoalSensor(MultiObjSensor):
    cls_uuid: str = "abs_obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        _, pos = self._sim.get_targets()
        return pos


@registry.register_sensor
class DummySensor(Sensor):
    def _get_uuid(self, *args, **kwargs):
        return "dummy"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        return np.zeros(1)


@registry.register_sensor
class LocalizationSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "localization"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        trans = self._sim.robot._robot.transformation
        forward = np.array([1.0, 0, 0])
        heading = np.array(trans.transform_vector(forward))
        forward = forward[[0, 2]]
        heading = heading[[0, 2]]

        heading_angle = get_angle(forward, heading)
        c = np.cross(forward, heading) < 0
        if not c:
            heading_angle = -1.0 * heading_angle
        return np.array([*trans.translation, heading_angle])


@registry.register_sensor
class JointSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(9,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.robot.get_arm_pos()
        return np.array(joints_pos).astype(np.float32)


@registry.register_sensor
class TrackMarkerSensor(Sensor):
    """
    Will track the first marker from the simulator's track markers array
    relative to the robot's EE position
    """

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "track_marker"

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
        pos = self._sim.get_track_markers_pos()[0]

        trans = self._sim.get_robot_transform()
        pos = trans.inverted().transform_point(pos)
        return np.array(pos).astype(np.float32)


@registry.register_sensor
class EeSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "ee_pos"

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
        trans = self._sim.robot._robot.transformation
        ee_pos = self._sim.robot.get_end_effector_transform().translation
        local_ee_pos = trans.inverted().transform_point(ee_pos)

        return np.array(local_ee_pos)


@registry.register_sensor
class IsHoldingSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "is_holding"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        snapped_id = self._sim.snapped_obj_id
        is_holding = (snapped_id is not None) or (
            self._sim.snapped_marker_name is not None
        )

        return np.array(int(is_holding)).reshape((1,))


@registry.register_measure
class ObjectToGoalDistance(Measure):
    cls_uuid: str = "object_to_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, episode, *args, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, episode, *args, **kwargs):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        distances = np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        self._metric = {idx: dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class EndEffectorToObjectDistance(Measure):
    cls_uuid: str = "ee_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, episode, *args, **kwargs):
        ee_pos = self._sim.robot.get_end_effector_transform().translation

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        distances = np.linalg.norm(target_pos - ee_pos, ord=2, axis=-1)

        self._metric = {idx: dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class DummyMeasure(Measure):
    cls_uuid: str = "dummy_measure"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DummyMeasure.cls_uuid

    def reset_metric(self, episode, *args, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, episode, *args, **kwargs):
        self._metric = 0


@registry.register_measure
class EndEffectorToPosDistance(Measure):
    cls_uuid: str = "ee_to_pos_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._target_pos = np.zeros(3)
        super().__init__(**kwargs)

    def set_target_pos(self, target_pos):
        self._target_pos = target_pos

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToPosDistance.cls_uuid

    def reset_metric(self, episode, *args, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, episode, *args, **kwargs):
        ee_pos = self._sim.get_end_effector_pos()

        distance = np.linalg.norm(self._target_pos - ee_pos, ord=2)

        self._metric = distance


@registry.register_measure
class RearrangePickReward(Measure):
    cls_uuid: str = "rearrangepick_reward"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickReward.cls_uuid

    def reset_metric(self, episode, task, observations, *args, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [EndEffectorToObjectDistance.cls_uuid]
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, episode, observations, task, *args, **kwargs):
        ee_to_object_distance = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()
        reward = 0

        snapped_id = self._sim.snapped_obj_id
        cur_picked = snapped_id is not None
        ee_pos = observations["ee_pos"]  # self._sim.get_end_effector_pos()  #

        if cur_picked:
            dist_to_goal = np.linalg.norm(ee_pos - task.desired_resting)
        else:
            dist_to_goal = ee_to_object_distance[task.targ_idx]

        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        did_pick = cur_picked and (not self.task)
        if did_pick:
            if snapped_id == abs_targ_obj_idx:
                task.n_succ_picks += 1
                reward += self._config.PICK_REWARD
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object...
                reward -= self._config.WRONG_PICK_PEN
                self.should_end = True
                self._metric = reward
                return

        if self._config.USE_DIFF:
            if self.cur_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self.cur_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            reward += self._config.DIST_REWARD * dist_diff
        else:
            reward -= self._config.DIST_REWARD * dist_to_goal
        self.cur_dist = dist_to_goal

        if not cur_picked and self.prev_picked:
            # Dropped the object...
            reward -= self._config.DROP_PEN
            self.should_end = True
            self._metric = reward
            return

        if self._my_episode_success():
            reward += self.rlcfg.SUCC_REWARD

        reward += self._get_coll_reward()

        self.prev_picked = cur_picked

        self._metric = reward
