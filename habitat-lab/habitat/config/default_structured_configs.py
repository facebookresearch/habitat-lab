#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import attr
from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING


@attr.s(auto_attribs=True, slots=True)
class HabitatBaseConfig:
    pass


@attr.s(auto_attribs=True, slots=True)
class IteratorOptionsConfig(HabitatBaseConfig):
    cycle: bool = True
    shuffle: bool = True
    group_by_scene: bool = True
    num_episode_sample: int = -1
    max_scene_repeat_episodes: int = -1
    max_scene_repeat_steps: int = int(1e4)
    step_repetition_range: float = 0.2


@attr.s(auto_attribs=True, slots=True)
class EnvironmentConfig(HabitatBaseConfig):
    max_episode_steps: int = 1000
    max_episode_seconds: int = 10000000
    iterator_options: IteratorOptionsConfig = IteratorOptionsConfig()


# -----------------------------------------------------------------------------
# # Actions
# -----------------------------------------------------------------------------
@attr.s(auto_attribs=True, slots=True)
class ActionConfig(HabitatBaseConfig):
    type: str = MISSING


@attr.s(auto_attribs=True, slots=True)
class StopActionConfig(ActionConfig):
    type: str = "StopAction"


@attr.s(auto_attribs=True, slots=True)
class EmptyActionConfig(ActionConfig):
    type: str = "EmptyAction"


# -----------------------------------------------------------------------------
# # NAVIGATION actions
# -----------------------------------------------------------------------------
@attr.s(auto_attribs=True, slots=True)
class MoveForwardActionConfig(ActionConfig):
    type: str = "MoveForwardAction"


@attr.s(auto_attribs=True, slots=True)
class TurnLeftActionConfig(ActionConfig):
    type: str = "TurnLeftAction"


@attr.s(auto_attribs=True, slots=True)
class TurnRightActionConfig(ActionConfig):
    type: str = "TurnRightAction"


@attr.s(auto_attribs=True, slots=True)
class LookUpActionConfig(ActionConfig):
    type: str = "LookUpAction"


@attr.s(auto_attribs=True, slots=True)
class LookDownActionConfig(ActionConfig):
    type: str = "LookDownAction"


@attr.s(auto_attribs=True, slots=True)
class TeleportActionConfig(ActionConfig):
    type: str = "TeleportAction"


@attr.s(auto_attribs=True, slots=True)
class VelocityControlActionConfig(ActionConfig):
    type: str = "VelocityAction"
    # meters/sec
    lin_vel_range: List[float] = [0.0, 0.25]
    # deg/sec
    ang_vel_range: List[float] = [-10.0, 10.0]
    min_abs_lin_speed: float = 0.025  # meters/sec
    min_abs_ang_speed: float = 1.0  # # deg/sec
    time_step: float = 1.0  # seconds


# -----------------------------------------------------------------------------
# # REARRANGE actions
# -----------------------------------------------------------------------------
@attr.s(auto_attribs=True, slots=True)
class ArmActionConfig(ActionConfig):
    r"""
    In Rearrangement tasks only, the action that will move the robot arm around. The action represents to delta angle (in radians) of each joint.

    :property grasp_thresh_dist: The grasp action will only work on the closest object if its distance to the end effector is smaller than this value. Only for `MagicGraspAction` grip_controller.
    :property grip_controller: Can either be None,  `MagicGraspAction` or `SuctionGraspAction`. If None, the arm will be unable to grip object. Magic grasp will grasp the object if the end effector is within grasp_thresh_dist of an object, with `SuctionGraspAction`, the object needs to be in contact with the end effector.
    :property gaze_distance_range: The gaze action will only work on the closet object if its distance to the end effector is smaller than this value. Only for `GazeGraspAction` grip_controller.
    :property center_cone_angle_threshold: The threshold angle between the line of sight and center_cone_vector. Only for `GazeGraspAction` grip_controller.
    :property center_cone_vector: The vector that the camera's line of sight should be when grasping the object. Only for `GazeGraspAction` grip_controller.
    :property grasp_threshold: The gripper is enabled if `grip_action` is greather than `grasp_threshold`
    """
    type: str = "ArmAction"
    arm_controller: str = "ArmRelPosAction"
    grip_controller: Optional[str] = None
    arm_joint_mask: Optional[List[int]] = None
    arm_joint_dimensionality: int = 7
    grasp_thresh_dist: float = 0.15
    disable_grip: bool = False
    max_delta_pos: float = (
        0.0125  # The maximum an arm joint can move in a single step
    )
    min_delta_pos: float = (
        0.0  # The minimum an arm joint needs to move in a single step
    )
    ee_ctrl_lim: float = 0.015
    should_clip: bool = False
    render_ee_target: bool = False
    gaze_distance_range: Optional[List[float]] = None
    center_cone_angle_threshold: float = 0.0
    center_cone_vector: Optional[List[float]] = None
    wrong_grasp_should_end: bool = False
    gaze_distance_from: str = "camera"
    gaze_center_square_width: float = 1
    grasp_threshold: float = 0.0
    oracle_snap: bool = False


@attr.s(auto_attribs=True, slots=True)
class BaseVelocityActionConfig(ActionConfig):
    type: str = "BaseVelAction"
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    allow_back: bool = True
    collision_threshold: float = 1e-5
    navmesh_offset: Optional[List[float]] = None
    min_displacement: float = 0.1  # minimum displacement that is allowed
    max_displacement_along_axis: float = 0.25  # maximum displacement
    max_turn_degrees: float = 30.0  # maximum turn waypoint
    min_turn_degrees: float = 5.0  # minimum turn waypoint
    allow_lateral_movement: bool = False  # whether to allow lateral movement
    allow_simultaneous_turn: bool = False  # whether to allow simultaneous turn
    discrete_movement: bool = (
        False  # whether to move/rotate only in discrete steps
    )
    constraint_base_in_manip_mode: bool = (
        False  # whether to constraint base motion in manip mode
    )


@attr.s(auto_attribs=True, slots=True)
class RearrangeStopActionConfig(ActionConfig):
    type: str = "RearrangeStopAction"


@attr.s(auto_attribs=True, slots=True)
class ManipulationModeActionConfig(ActionConfig):
    type: str = "ManipulationModeAction"
    threshold: float = 0.8


@attr.s(auto_attribs=True, slots=True)
class OracleNavActionConfig(ActionConfig):
    """
    Oracle navigation action.
    This action takes as input a discrete ID which refers to an object in the
    PDDL domain. The oracle navigation controller then computes the actions to
    navigate to that desired object.
    """

    type: str = "OracleNavAction"
    turn_velocity: float = 1.0
    forward_velocity: float = 1.0
    turn_thresh: float = 0.1
    dist_thresh: float = 0.2
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    allow_back: bool = True


# -----------------------------------------------------------------------------
# # EQA actions
# -----------------------------------------------------------------------------
@attr.s(auto_attribs=True, slots=True)
class AnswerActionConfig(ActionConfig):
    type: str = "AnswerAction"


# -----------------------------------------------------------------------------
# # TASK_SENSORS
# -----------------------------------------------------------------------------
@attr.s(auto_attribs=True, slots=True)
class LabSensorConfig(HabitatBaseConfig):
    type: str = MISSING


@attr.s(auto_attribs=True, slots=True)
class PointGoalSensorConfig(LabSensorConfig):
    type: str = "PointGoalSensor"
    goal_format: str = "POLAR"
    dimensionality: int = 2


@attr.s(auto_attribs=True, slots=True)
class PointGoalWithGPSCompassSensorConfig(PointGoalSensorConfig):
    type: str = "PointGoalWithGPSCompassSensor"


@attr.s(auto_attribs=True, slots=True)
class ObjectGoalSensorConfig(LabSensorConfig):
    type: str = "ObjectGoalSensor"
    goal_spec: str = "TASK_CATEGORY_ID"
    goal_spec_max_val: int = 50


@attr.s(auto_attribs=True, slots=True)
class ImageGoalSensorConfig(LabSensorConfig):
    type: str = "ImageGoalSensor"


@attr.s(auto_attribs=True, slots=True)
class InstanceImageGoalSensorConfig(LabSensorConfig):
    type: str = "InstanceImageGoalSensor"


@attr.s(auto_attribs=True, slots=True)
class InstanceImageGoalHFOVSensorConfig(LabSensorConfig):
    type: str = "InstanceImageGoalHFOVSensor"


@attr.s(auto_attribs=True, slots=True)
class HeadingSensorConfig(LabSensorConfig):
    type: str = "HeadingSensor"


@attr.s(auto_attribs=True, slots=True)
class CompassSensorConfig(LabSensorConfig):
    type: str = "CompassSensor"


@attr.s(auto_attribs=True, slots=True)
class GPSSensorConfig(LabSensorConfig):
    type: str = "GPSSensor"
    dimensionality: int = 2


@attr.s(auto_attribs=True, slots=True)
class RobotStartCompassSensorConfig(CompassSensorConfig):
    type: str = "RobotStartCompassSensor"


@attr.s(auto_attribs=True, slots=True)
class RobotStartGPSSensorConfig(GPSSensorConfig):
    type: str = "RobotStartGPSSensor"


@attr.s(auto_attribs=True, slots=True)
class ProximitySensorConfig(LabSensorConfig):
    type: str = "ProximitySensor"
    max_detection_radius: float = 2.0


@attr.s(auto_attribs=True, slots=True)
class JointSensorConfig(LabSensorConfig):
    type: str = "JointSensor"
    dimensionality: int = 7


@attr.s(auto_attribs=True, slots=True)
class EEPositionSensorConfig(LabSensorConfig):
    type: str = "EEPositionSensor"


@attr.s(auto_attribs=True, slots=True)
class IsHoldingSensorConfig(LabSensorConfig):
    type: str = "IsHoldingSensor"


@attr.s(auto_attribs=True, slots=True)
class RelativeRestingPositionSensorConfig(LabSensorConfig):
    type: str = "RelativeRestingPositionSensor"


@attr.s(auto_attribs=True, slots=True)
class JointVelocitySensorConfig(LabSensorConfig):
    type: str = "JointVelocitySensor"
    dimensionality: int = 7


@attr.s(auto_attribs=True, slots=True)
class OracleNavigationActionSensorConfig(LabSensorConfig):
    type: str = "OracleNavigationActionSensor"


@attr.s(auto_attribs=True, slots=True)
class RestingPositionSensorConfig(LabSensorConfig):
    type: str = "RestingPositionSensor"


@attr.s(auto_attribs=True, slots=True)
class ArtJointSensorConfig(LabSensorConfig):
    type: str = "ArtJointSensor"


@attr.s(auto_attribs=True, slots=True)
class NavGoalSensorConfig(LabSensorConfig):
    type: str = "NavGoalSensor"


@attr.s(auto_attribs=True, slots=True)
class ArtJointSensorNoVelSensorConfig(LabSensorConfig):
    type: str = "ArtJointSensorNoVel"  # TODO: add "Sensor" suffix


@attr.s(auto_attribs=True, slots=True)
class MarkerRelPosSensorConfig(LabSensorConfig):
    type: str = "MarkerRelPosSensor"


@attr.s(auto_attribs=True, slots=True)
class TargetStartSensorConfig(LabSensorConfig):
    type: str = "TargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@attr.s(auto_attribs=True, slots=True)
class TargetCurrentSensorConfig(LabSensorConfig):
    type: str = "TargetCurrentSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@attr.s(auto_attribs=True, slots=True)
class GoalSensorConfig(LabSensorConfig):
    type: str = "GoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@attr.s(auto_attribs=True, slots=True)
class NavGoalPointGoalSensorConfig(LabSensorConfig):
    type: str = "NavGoalPointGoalSensor"


@attr.s(auto_attribs=True, slots=True)
class GlobalPredicatesSensorConfig(LabSensorConfig):
    type: str = "GlobalPredicatesSensor"


@attr.s(auto_attribs=True, slots=True)
class TargetStartGpsCompassSensorConfig(LabSensorConfig):
    type: str = "TargetStartGpsCompassSensor"


@attr.s(auto_attribs=True, slots=True)
class TargetGoalGpsCompassSensorConfig(LabSensorConfig):
    type: str = "TargetGoalGpsCompassSensor"


@attr.s(auto_attribs=True, slots=True)
class NavToSkillSensorConfig(LabSensorConfig):
    type: str = "NavToSkillSensor"
    num_skills: int = 8


@attr.s(auto_attribs=True, slots=True)
class AbsTargetStartSensorConfig(LabSensorConfig):
    type: str = "AbsTargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@attr.s(auto_attribs=True, slots=True)
class AbsGoalSensorConfig(LabSensorConfig):
    type: str = "AbsGoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@attr.s(auto_attribs=True, slots=True)
class DistToNavGoalSensorConfig(LabSensorConfig):
    type: str = "DistToNavGoalSensor"


@attr.s(auto_attribs=True, slots=True)
class ObjectCategorySensorConfig(LabSensorConfig):
    type: str = "ObjectCategorySensor"


@attr.s(auto_attribs=True, slots=True)
class GoalReceptacleSensorConfig(LabSensorConfig):
    type: str = "GoalReceptacleSensor"


@attr.s(auto_attribs=True, slots=True)
class StartReceptacleSensorConfig(LabSensorConfig):
    type: str = "StartReceptacleSensor"


@attr.s(auto_attribs=True, slots=True)
class ObjectEmbeddingSensorConfig(LabSensorConfig):
    type: str = "ObjectEmbeddingSensor"
    embeddings_file: str = "data/objects/clip_embeddings.pickle"
    dimensionality: int = 512


@attr.s(auto_attribs=True, slots=True)
class ObjectSegmentationSensorConfig(LabSensorConfig):
    type: str = "ObjectSegmentationSensor"
    blank_out_prob: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class StartRecepSegmentationSensorConfig(ObjectSegmentationSensorConfig):
    type: str = "StartRecepSegmentationSensor"


@attr.s(auto_attribs=True, slots=True)
class GoalRecepSegmentationSensorConfig(ObjectSegmentationSensorConfig):
    type: str = "GoalRecepSegmentationSensor"


@attr.s(auto_attribs=True, slots=True)
class CameraPoseSensorConfig(LabSensorConfig):
    type: str = "CameraPoseSensor"


@attr.s(auto_attribs=True, slots=True)
class ReceptacleSegmentationSensorConfig(LabSensorConfig):
    type: str = "ReceptacleSegmentationSensor"
    blank_out_prob: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class OVMMNavGoalSegmentationSensorConfig(LabSensorConfig):
    type: str = "OVMMNavGoalSegmentationSensor"
    blank_out_prob: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class LocalizationSensorConfig(LabSensorConfig):
    type: str = "LocalizationSensor"


@attr.s(auto_attribs=True, slots=True)
class QuestionSensorConfig(LabSensorConfig):
    type: str = "QuestionSensor"


@attr.s(auto_attribs=True, slots=True)
class InstructionSensorConfig(LabSensorConfig):
    type: str = "InstructionSensor"
    instruction_sensor_uuid: str = "instruction"


# -----------------------------------------------------------------------------
# Measurements
# -----------------------------------------------------------------------------
@attr.s(auto_attribs=True, slots=True)
class MeasurementConfig(HabitatBaseConfig):
    type: str = MISSING


@attr.s(auto_attribs=True, slots=True)
class SuccessMeasurementConfig(MeasurementConfig):
    type: str = "Success"
    success_distance: float = 0.2


@attr.s(auto_attribs=True, slots=True)
class SPLMeasurementConfig(MeasurementConfig):
    type: str = "SPL"


@attr.s(auto_attribs=True, slots=True)
class SoftSPLMeasurementConfig(MeasurementConfig):
    type: str = "SoftSPL"


@attr.s(auto_attribs=True, slots=True)
class FogOfWarConfig:
    draw: bool = True
    visibility_dist: float = 5.0
    fov: int = 90


@attr.s(auto_attribs=True, slots=True)
class TopDownMapMeasurementConfig(MeasurementConfig):
    type: str = "TopDownMap"
    max_episode_steps: int = (
        EnvironmentConfig().max_episode_steps
    )  # TODO : Use OmegaConf II()
    map_padding: int = 3
    map_resolution: int = 1024
    draw_source: bool = True
    draw_border: bool = True
    draw_shortest_path: bool = True
    draw_view_points: bool = True
    draw_goal_positions: bool = True
    # axes aligned bounding boxes
    draw_goal_aabbs: bool = True
    fog_of_war: FogOfWarConfig = FogOfWarConfig()


@attr.s(auto_attribs=True, slots=True)
class CollisionsMeasurementConfig(MeasurementConfig):
    type: str = "Collisions"


@attr.s(auto_attribs=True, slots=True)
class RobotForceMeasurementConfig(MeasurementConfig):
    type: str = "RobotForce"
    min_force: float = 20.0


@attr.s(auto_attribs=True, slots=True)
class ForceTerminateMeasurementConfig(MeasurementConfig):
    type: str = "ForceTerminate"
    max_accum_force: float = -1.0
    max_instant_force: float = -1.0


@attr.s(auto_attribs=True, slots=True)
class RobotCollisionsTerminateMeasurementConfig(MeasurementConfig):
    type: str = "RobotCollisionsTerminate"
    max_num_collisions: int = -1  # do not terminate by default


@attr.s(auto_attribs=True, slots=True)
class RobotCollisionsMeasurementConfig(MeasurementConfig):
    type: str = "RobotCollisions"


@attr.s(auto_attribs=True, slots=True)
class ObjectToGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "ObjectToGoalDistance"


@attr.s(auto_attribs=True, slots=True)
class EndEffectorToObjectDistanceMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorToObjectDistance"


@attr.s(auto_attribs=True, slots=True)
class EndEffectorToRestDistanceMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorToRestDistance"


@attr.s(auto_attribs=True, slots=True)
class EndEffectorToGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorToGoalDistance"


@attr.s(auto_attribs=True, slots=True)
class ArtObjAtDesiredStateMeasurementConfig(MeasurementConfig):
    type: str = "ArtObjAtDesiredState"
    use_absolute_distance: bool = True
    success_dist_threshold: float = 0.05


@attr.s(auto_attribs=True, slots=True)
class GfxReplayMeasureMeasurementConfig(MeasurementConfig):
    type: str = "GfxReplayMeasure"


@attr.s(auto_attribs=True, slots=True)
class EndEffectorDistToMarkerMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorDistToMarker"


@attr.s(auto_attribs=True, slots=True)
class ArtObjStateMeasurementConfig(MeasurementConfig):
    type: str = "ArtObjState"


@attr.s(auto_attribs=True, slots=True)
class ArtObjSuccessMeasurementConfig(MeasurementConfig):
    type: str = "ArtObjSuccess"
    rest_dist_threshold: float = 0.15
    must_call_stop: bool = True


@attr.s(auto_attribs=True, slots=True)
class ArtObjRewardMeasurementConfig(MeasurementConfig):
    type: str = "ArtObjReward"
    dist_reward: float = 1.0
    wrong_grasp_end: bool = False
    wrong_grasp_pen: float = 5.0
    art_dist_reward: float = 10.0
    ee_dist_reward: float = 10.0
    marker_dist_reward: float = 0.0
    art_at_desired_state_reward: float = 5.0
    grasp_reward: float = 0.0
    # General Rearrange Reward config
    constraint_violate_pen: float = 10.0
    navmesh_violate_pen: float = (
        0.0  # penalty for trying to move outside navmesh
    )
    force_pen: float = 0.0
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0
    robot_collisions_pen: float = 0.0
    robot_collisions_end_pen: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class RotDistToGoalMeasurementConfig(MeasurementConfig):
    type: str = "RotDistToGoal"


@attr.s(auto_attribs=True, slots=True)
class NavmeshCollisionMeasurementConfig(MeasurementConfig):
    type: str = "NavmeshCollision"


@attr.s(auto_attribs=True, slots=True)
class DistToGoalMeasurementConfig(MeasurementConfig):
    type: str = "DistToGoal"
    use_shortest_path_cache: bool = True


@attr.s(auto_attribs=True, slots=True)
class BadCalledTerminateMeasurementConfig(MeasurementConfig):
    type: str = "BadCalledTerminate"
    bad_term_pen: float = 0.0
    decay_bad_term: bool = False


@attr.s(auto_attribs=True, slots=True)
class NavToPosSuccMeasurementConfig(MeasurementConfig):
    type: str = "NavToPosSucc"
    success_distance: float = 0.5


@attr.s(auto_attribs=True, slots=True)
class NavToObjRewardMeasurementConfig(MeasurementConfig):
    type: str = "NavToObjReward"
    # reward the agent for facing the object?
    should_reward_turn: bool = True
    # what distance do we start giving the reward for facing the object?
    turn_reward_dist: float = 3.0
    # multiplier on the angle distance to the goal.
    angle_dist_reward: float = 1.0
    dist_reward: float = 1.0
    constraint_violate_pen: float = 1.0
    navmesh_violate_pen: float = (
        0.0  # penalty for trying to move outside navmesh
    )
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    robot_collisions_pen: float = 0.0
    robot_collisions_end_pen: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class NavToObjSuccessMeasurementConfig(MeasurementConfig):
    type: str = "NavToObjSuccess"
    must_look_at_targ: bool = True
    must_call_stop: bool = True
    # distance in radians.
    success_angle_dist: float = 0.261799


@attr.s(auto_attribs=True, slots=True)
class OVMMNavToObjRewardMeasurementConfig(NavToObjRewardMeasurementConfig):
    type: str = "OVMMNavToObjReward"


@attr.s(auto_attribs=True, slots=True)
class OVMMDistToPickGoalMeasurementConfig(DistToGoalMeasurementConfig):
    type: str = "OVMMDistToPickGoal"


@attr.s(auto_attribs=True, slots=True)
class OVMMDistToPlaceGoalMeasurementConfig(DistToGoalMeasurementConfig):
    type: str = "OVMMDistToPlaceGoal"


@attr.s(auto_attribs=True, slots=True)
class OVMMRotDistToGoalMeasurementConfig(RotDistToGoalMeasurementConfig):
    type: str = "OVMMRotDistToGoal"


@attr.s(auto_attribs=True, slots=True)
class OVMMRotDistToPickGoalMeasurementConfig(
    OVMMRotDistToGoalMeasurementConfig
):
    type: str = "OVMMRotDistToPickGoal"


@attr.s(auto_attribs=True, slots=True)
class OVMMRotDistToPlaceGoalMeasurementConfig(
    OVMMRotDistToGoalMeasurementConfig
):
    type: str = "OVMMRotDistToPlaceGoal"


@attr.s(auto_attribs=True, slots=True)
class OVMMNavToPickSuccMeasurementConfig(NavToPosSuccMeasurementConfig):
    type: str = "OVMMNavToPickSucc"


@attr.s(auto_attribs=True, slots=True)
class OVMMNavToObjSuccMeasurementConfig(NavToObjSuccessMeasurementConfig):
    type: str = "OVMMNavToObjSucc"
    min_object_coverage_iou: float = 1e-3


@attr.s(auto_attribs=True, slots=True)
class OVMMNavOrientToPickSuccMeasurementConfig(
    OVMMNavToObjSuccMeasurementConfig
):
    type: str = "OVMMNavOrientToPickSucc"


@attr.s(auto_attribs=True, slots=True)
class OVMMNavToPlaceSuccMeasurementConfig(NavToPosSuccMeasurementConfig):
    type: str = "OVMMNavToPlaceSucc"


@attr.s(auto_attribs=True, slots=True)
class OVMMNavOrientToPlaceSuccMeasurementConfig(
    OVMMNavToObjSuccMeasurementConfig
):
    type: str = "OVMMNavOrientToPlaceSucc"


@attr.s(auto_attribs=True, slots=True)
class PickObjectIoUCoverageMeasurementConfig(MeasurementConfig):
    type: str = "PickObjectIoUCoverage"


@attr.s(auto_attribs=True, slots=True)
class PlaceObjectIoUCoverageMeasurementConfig(MeasurementConfig):
    type: str = "PlaceObjectIoUCoverage"


@attr.s(auto_attribs=True, slots=True)
class TargetIoUCoverageMeasurementConfig(MeasurementConfig):
    type: str = "TargetIoUCoverage"
    max_goal_dist: float = 0.1


@attr.s(auto_attribs=True, slots=True)
class PickGoalIoUCoverageMeasurementConfig(TargetIoUCoverageMeasurementConfig):
    type: str = "PickGoalIoUCoverage"


@attr.s(auto_attribs=True, slots=True)
class PlaceGoalIoUCoverageMeasurementConfig(
    TargetIoUCoverageMeasurementConfig
):
    type: str = "PlaceGoalIoUCoverage"


@attr.s(auto_attribs=True, slots=True)
class RearrangeReachRewardMeasurementConfig(MeasurementConfig):
    type: str = "RearrangeReachReward"
    scale: float = 1.0
    diff_reward: bool = True
    sparse_reward: bool = False


@attr.s(auto_attribs=True, slots=True)
class RearrangeReachSuccessMeasurementConfig(MeasurementConfig):
    type: str = "RearrangeReachSuccess"
    succ_thresh: float = 0.2


@attr.s(auto_attribs=True, slots=True)
class NumStepsMeasurementConfig(MeasurementConfig):
    type: str = "NumStepsMeasure"


@attr.s(auto_attribs=True, slots=True)
class DidPickObjectMeasurementConfig(MeasurementConfig):
    type: str = "DidPickObjectMeasure"


@attr.s(auto_attribs=True, slots=True)
class DidViolateHoldConstraintMeasurementConfig(MeasurementConfig):
    type: str = "DidViolateHoldConstraintMeasure"


@attr.s(auto_attribs=True, slots=True)
class MoveObjectsRewardMeasurementConfig(MeasurementConfig):
    type: str = "MoveObjectsReward"
    pick_reward: float = 1.0
    success_dist: float = 0.15
    single_rearrange_reward: float = 1.0
    dist_reward: float = 1.0
    constraint_violate_pen: float = 10.0
    navmesh_violate_pen: float = (
        0.0  # penalty for trying to move outside navmesh
    )
    force_pen: float = 0.001
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0
    robot_collisions_pen: float = 0.0
    robot_collisions_end_pen: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class RearrangePickRewardMeasurementConfig(MeasurementConfig):
    type: str = "RearrangePickReward"
    dist_reward: float = 2.0
    pick_reward: float = 2.0
    constraint_violate_pen: float = 1.0
    navmesh_violate_pen: float = (
        0.0  # penalty for trying to move outside navmesh
    )
    drop_pen: float = 0.5
    wrong_pick_pen: float = 0.5
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    use_diff: bool = True
    drop_obj_should_end: bool = True
    wrong_pick_should_end: bool = True
    object_goal: bool = False
    sparse_reward: bool = False
    angle_reward_min_dist: float = 0.0
    angle_reward_scale: float = 1.0
    robot_collisions_pen: float = 0.0
    robot_collisions_end_pen: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class RearrangePickSuccessMeasurementConfig(MeasurementConfig):
    type: str = "RearrangePickSuccess"
    ee_resting_success_threshold: float = 0.15
    object_goal: bool = False


@attr.s(auto_attribs=True, slots=True)
class ObjAtGoalMeasurementConfig(MeasurementConfig):
    type: str = "ObjAtGoal"
    succ_thresh: float = 0.15


@attr.s(auto_attribs=True, slots=True)
class ObjAnywhereOnGoalMeasurementConfig(MeasurementConfig):
    type: str = "ObjAnywhereOnGoal"


@attr.s(auto_attribs=True, slots=True)
class PlaceRewardMeasurementConfig(MeasurementConfig):
    type: str = "PlaceReward"
    dist_reward: float = 2.0
    place_reward: float = 5.0
    drop_pen: float = 0.0
    use_diff: bool = True
    use_ee_dist: bool = False
    wrong_drop_should_end: bool = True
    constraint_violate_pen: float = 0.0
    navmesh_violate_pen: float = (
        0.0  # penalty for trying to move outside navmesh
    )
    force_pen: float = 0.0001
    max_force_pen: float = 0.0
    force_end_pen: float = 1.0
    min_dist_to_goal: float = 0.15
    sparse_reward: bool = False
    drop_pen_type: str = "constant"
    ee_resting_success_threshold: float = 0.15
    stability_reward: float = 0.0
    max_steps_to_reach_surface: int = 0
    robot_collisions_pen: float = 0.0
    robot_collisions_end_pen: float = 0.0


@attr.s(auto_attribs=True, slots=True)
class PlacementStabilityMeasurementConfig(MeasurementConfig):
    type: str = "PlacementStability"
    stability_steps: float = 50


@attr.s(auto_attribs=True, slots=True)
class PlaceSuccessMeasurementConfig(MeasurementConfig):
    type: str = "PlaceSuccess"
    ee_resting_success_threshold: float = 0.15
    check_stability: bool = False


@attr.s(auto_attribs=True, slots=True)
class PickedObjectLinearVelMeasurementConfig(MeasurementConfig):
    type: str = "PickedObjectLinearVel"


@attr.s(auto_attribs=True, slots=True)
class PickedObjectAngularVelMeasurementConfig(MeasurementConfig):
    type: str = "PickedObjectAngularVel"


@attr.s(auto_attribs=True, slots=True)
class ObjectAtRestMeasurementConfig(MeasurementConfig):
    type: str = "ObjectAtRest"
    angular_vel_thresh: float = 1e-3
    linear_vel_thresh: float = 1e-3


@attr.s(auto_attribs=True, slots=True)
class OVMMFindObjectPhaseSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OVMMFindObjectPhaseSuccess"


@attr.s(auto_attribs=True, slots=True)
class OVMMPickObjectPhaseSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OVMMPickObjectPhaseSuccess"


@attr.s(auto_attribs=True, slots=True)
class OVMMPlaceObjectPhaseSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OVMMPlaceObjectPhaseSuccess"


@attr.s(auto_attribs=True, slots=True)
class OVMMFindRecepPhaseSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OVMMFindRecepPhaseSuccess"


@attr.s(auto_attribs=True, slots=True)
class OVMMObjectToPlaceGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "OVMMObjectToPlaceGoalDistance"


@attr.s(auto_attribs=True, slots=True)
class OVMMEEToPlaceGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "OVMMEEToPlaceGoalDistance"


@attr.s(auto_attribs=True, slots=True)
class OVMMPlaceRewardMeasurementConfig(PlaceRewardMeasurementConfig):
    type: str = "OVMMPlaceReward"


@attr.s(auto_attribs=True, slots=True)
class OVMMPlacementStabilityMeasurementConfig(
    PlacementStabilityMeasurementConfig
):
    type: str = "OVMMPlacementStability"


@attr.s(auto_attribs=True, slots=True)
class OVMMPlaceSuccessMeasurementConfig(PlaceSuccessMeasurementConfig):
    type: str = "OVMMPlaceSuccess"


@attr.s(auto_attribs=True, slots=True)
class CompositeNodeIdxMeasurementConfig(MeasurementConfig):
    type: str = "CompositeNodeIdx"


@attr.s(auto_attribs=True, slots=True)
class CompositeStageGoalsMeasurementConfig(MeasurementConfig):
    type: str = "CompositeStageGoals"


@attr.s(auto_attribs=True, slots=True)
class CompositeSuccessMeasurementConfig(MeasurementConfig):
    type: str = "CompositeSuccess"
    must_call_stop: bool = True


@attr.s(auto_attribs=True, slots=True)
class CompositeRewardMeasurementConfig(MeasurementConfig):
    type: str = "CompositeReward"
    must_call_stop: bool = True
    success_reward: float = 10.0


@attr.s(auto_attribs=True, slots=True)
class DoesWantTerminateMeasurementConfig(MeasurementConfig):
    type: str = "DoesWantTerminate"


@attr.s(auto_attribs=True, slots=True)
class CorrectAnswerMeasurementConfig(MeasurementConfig):
    type: str = "CorrectAnswer"


@attr.s(auto_attribs=True, slots=True)
class EpisodeInfoMeasurementConfig(MeasurementConfig):
    type: str = "EpisodeInfo"


@attr.s(auto_attribs=True, slots=True)
class DistanceToGoalMeasurementConfig(MeasurementConfig):
    type: str = "DistanceToGoal"
    distance_to: str = "POINT"
    goals_attr: str = "goals"
    distance_from: str = "BASE"


@attr.s(auto_attribs=True, slots=True)
class DistanceToGoalRewardMeasurementConfig(MeasurementConfig):
    type: str = "DistanceToGoalReward"


@attr.s(auto_attribs=True, slots=True)
class PickDistanceToGoalMeasurementConfig(DistanceToGoalMeasurementConfig):
    type: str = "PickDistanceToGoal"


@attr.s(auto_attribs=True, slots=True)
class PickDistanceToGoalRewardMeasurementConfig(MeasurementConfig):
    type: str = "PickDistanceToGoalReward"


@attr.s(auto_attribs=True, slots=True)
class AnswerAccuracyMeasurementConfig(MeasurementConfig):
    type: str = "AnswerAccuracy"


@attr.s(auto_attribs=True, slots=True)
class TaskConfig(HabitatBaseConfig):
    reward_measure: Optional[str] = None
    success_measure: Optional[str] = None
    success_reward: float = 2.5
    slack_reward: float = -0.01
    end_on_success: bool = False
    # NAVIGATION task
    type: str = "Nav-v0"
    # Temporary structure for sensors
    lab_sensors: Dict[str, LabSensorConfig] = dict()
    measurements: Dict[str, MeasurementConfig] = dict()
    goal_sensor_uuid: str = "pointgoal"
    # REARRANGE task
    count_obj_collisions: bool = True
    settle_steps: int = 5
    constraint_violation_ends_episode: bool = True
    constraint_violation_drops_object: bool = False
    # Forced to regenerate the starts even if they are already cached
    force_regenerate: bool = False
    # Saves the generated starts to a cache if they are not already generated
    should_save_to_cache: bool = False
    must_look_at_targ: bool = True
    object_in_hand_sample_prob: float = 0.167
    min_start_distance: float = 3.0
    gfx_replay_dir = "data/replays"
    render_target: bool = True
    # Spawn parameters
    physics_stability_steps: int = 1
    num_spawn_attempts: int = 200
    spawn_max_dists_to_obj: float = 1.0
    base_angle_noise: float = 0.523599
    spawn_reference: str = "target"
    spawn_reference_sampling: str = "uniform"
    # EE sample parameters
    ee_sample_factor: float = 0.2
    ee_exclude_region: float = 0.0
    base_noise: float = 0.05
    spawn_region_scale: float = 0.2
    joint_max_impulse: float = -1.0
    desired_resting_position: List[float] = [0.5, 0.0, 1.0]
    use_marker_t: bool = True
    cache_robot_init: bool = False
    success_state: float = 0.0
    # Measurements for composite tasks.
    # If true, does not care about navigability or collisions
    # with objects when spawning robot
    easy_init: bool = False
    should_enforce_target_within_reach: bool = False
    # COMPOSITE task CONFIG
    task_spec_base_path: str = "habitat/task/rearrange/pddl/"
    task_spec: str = ""
    # PDDL domain params
    pddl_domain_def: str = "replica_cad"
    obj_succ_thresh: float = 0.3
    # Disable drop except for when the object is at its goal.
    enable_safe_drop: bool = False
    art_succ_thresh: float = 0.15
    robot_at_thresh: float = 2.0
    filter_nav_to_tasks: List = []
    actions: Dict[str, ActionConfig] = MISSING
    start_in_manip_mode: bool = False
    pick_init: bool = False
    place_init: bool = False
    episode_init: bool = False
    camera_tilt: float = -0.5236
    receptacle_categories_file: str = ""


@attr.s(auto_attribs=True, slots=True)
class SimulatorSensorConfig(HabitatBaseConfig):
    type: str = MISSING
    height: int = 480
    width: int = 640
    position: List[float] = [0.0, 1.25, 0.0]
    # Euler's angles:
    orientation: List[float] = [0.0, 0.0, 0.0]


@attr.s(auto_attribs=True, slots=True)
class SimulatorCameraSensorConfig(SimulatorSensorConfig):
    hfov: int = 90  # horizontal field of view in degrees
    sensor_subtype: str = "PINHOLE"
    noise_model: str = "None"
    noise_model_kwargs: Dict[str, Any] = dict()


@attr.s(auto_attribs=True, slots=True)
class SimulatorDepthSensorConfig(SimulatorSensorConfig):
    min_depth: float = 0.0
    max_depth: float = 10.0
    normalize_depth: bool = True


@attr.s(auto_attribs=True, slots=True)
class HabitatSimRGBSensorConfig(SimulatorCameraSensorConfig):
    type: str = "HabitatSimRGBSensor"


@attr.s(auto_attribs=True, slots=True)
class HabitatSimDepthSensorConfig(SimulatorCameraSensorConfig):
    type: str = "HabitatSimDepthSensor"
    min_depth: float = 0.0
    max_depth: float = 10.0
    normalize_depth: bool = True


@attr.s(auto_attribs=True, slots=True)
class HabitatSimSemanticSensorConfig(SimulatorCameraSensorConfig):
    type: str = "HabitatSimSemanticSensor"


@attr.s(auto_attribs=True, slots=True)
class HabitatSimEquirectangularRGBSensorConfig(SimulatorSensorConfig):
    type: str = "HabitatSimEquirectangularRGBSensor"


@attr.s(auto_attribs=True, slots=True)
class HabitatSimEquirectangularDepthSensorConfig(SimulatorDepthSensorConfig):
    type: str = "HabitatSimEquirectangularDepthSensor"


@attr.s(auto_attribs=True, slots=True)
class HabitatSimEquirectangularSemanticSensorConfig(SimulatorSensorConfig):
    type: str = "HabitatSimEquirectangularSemanticSensor"


@attr.s(auto_attribs=True, slots=True)
class SimulatorFisheyeSensorConfig(SimulatorSensorConfig):
    type: str = "HabitatSimFisheyeSensor"
    height: int = SimulatorSensorConfig().width
    # The default value (alpha, xi) is set to match the lens  "GoPro" found in
    # Table 3 of this paper: Vladyslav Usenko, Nikolaus Demmel and
    # Daniel Cremers: The Double Sphere Camera Model,
    # The International Conference on 3D Vision (3DV), 2018
    # You can find the intrinsic parameters for the other lenses
    # in the same table as well.
    xi: float = -0.27
    alpha: float = 0.57
    focal_length: List[float] = [364.84, 364.86]
    # Place camera at center of screen
    # Can be specified, otherwise is calculated automatically.
    # principal_point_offset defaults to (h/2,w/2)
    principal_point_offset: Optional[List[float]] = None
    sensor_model_type: str = "DOUBLE_SPHERE"


@attr.s(auto_attribs=True, slots=True)
class HabitatSimFisheyeRGBSensorConfig(SimulatorFisheyeSensorConfig):
    type: str = "HabitatSimFisheyeRGBSensor"


@attr.s(auto_attribs=True, slots=True)
class SimulatorFisheyeDepthSensorConfig(SimulatorFisheyeSensorConfig):
    type: str = "HabitatSimFisheyeDepthSensor"
    min_depth: float = SimulatorDepthSensorConfig().min_depth
    max_depth: float = SimulatorDepthSensorConfig().max_depth
    normalize_depth: bool = SimulatorDepthSensorConfig().normalize_depth


@attr.s(auto_attribs=True, slots=True)
class HabitatSimFisheyeSemanticSensorConfig(SimulatorFisheyeSensorConfig):
    type: str = "HabitatSimFisheyeSemanticSensor"


@attr.s(auto_attribs=True, slots=True)
class HeadRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "robot_head_rgb"
    width: int = 256
    height: int = 256


@attr.s(auto_attribs=True, slots=True)
class HeadDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "robot_head_depth"
    width: int = 256
    height: int = 256


@attr.s(auto_attribs=True, slots=True)
class HeadPanopticSensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "robot_head_panoptic"
    width: int = 256
    height: int = 256


@attr.s(auto_attribs=True, slots=True)
class ArmRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "robot_arm_rgb"
    width: int = 256
    height: int = 256


@attr.s(auto_attribs=True, slots=True)
class ArmDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "robot_arm_depth"
    width: int = 256
    height: int = 256


@attr.s(auto_attribs=True, slots=True)
class ThirdRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "robot_third_rgb"
    width: int = 512
    height: int = 512


@attr.s(auto_attribs=True, slots=True)
class ThirdDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "robot_third_depth"  # TODO: robot_third_rgb on the main branch
    #  check if it won't cause any errors


@attr.s(auto_attribs=True, slots=True)
class AgentConfig(HabitatBaseConfig):
    height: float = 1.5
    radius: float = 0.1
    sim_sensors: Dict[str, SimulatorSensorConfig] = dict()
    is_set_start_state: bool = False
    start_position: List[float] = [0, 0, 0]
    start_rotation: List[float] = [0, 0, 0, 1]
    joint_start_noise: float = 0.1
    robot_urdf: str = "data/robots/hab_fetch/robots/hab_fetch.urdf"
    robot_type: str = "FetchRobot"
    ik_arm_urdf: Optional[
        str
    ] = "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"


@attr.s(auto_attribs=True, slots=True)
class HabitatSimV0Config(HabitatBaseConfig):
    gpu_device_id: int = 0
    # Use Habitat-Sim's GPU->GPU copy mode to return rendering results in
    # pytorch tensors. Requires Habitat-Sim to be built with --with-cuda.
    # This will generally imply sharing cuda tensors between processes.
    # Read here:
    # https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # for the caveats that results in
    gpu_gpu: bool = False
    # Whether the agent slides on collisions
    allow_sliding: bool = True
    frustum_culling: bool = True
    enable_physics: bool = False
    physics_config_file: str = "./data/default.physics_config.json"
    # Possibly unstable optimization for extra performance
    # with concurrent rendering
    leave_context_with_background_renderer: bool = False
    enable_gfx_replay_save: bool = False
    hbao_visual_effects: bool = False
    pbr_image_based_lighting: bool = True


@attr.s(auto_attribs=True, slots=True)
class SimulatorConfig(HabitatBaseConfig):
    type: str = "Sim-v0"
    action_space_config: str = "v0"
    action_space_config_arguments: Dict[str, Any] = dict()
    forward_step_size: float = 0.25  # in metres
    create_renderer: bool = False
    requires_textures: bool = True
    auto_sleep: bool = False
    sleep_dist: float = 3.0
    step_physics: bool = True
    concur_render: bool = False
    # If markers should be updated at every step:
    needs_markers: bool = True
    # If the robot camera positions should be updated at every step:
    update_robot: bool = True
    scene: str = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    # The scene dataset to load in the metadatamediator,
    # should contain simulator.scene:
    scene_dataset: str = "default"
    # A list of directory or config paths to search in addition to the dataset
    # for object configs. should match the generated episodes for the task:
    additional_object_paths: List[str] = []
    # Use config.seed (can't reference Config.seed) or define via code
    # otherwise it leads to circular references:
    #
    seed: int = II("habitat.seed")
    turn_angle: int = 10  # angle to rotate left or right in degrees
    tilt_angle: int = 15  # angle to tilt the camera up or down in degrees
    default_agent_id: int = 0
    debug_render: bool = False
    debug_render_robot: bool = False
    kinematic_mode: bool = False
    # If in render mode a visualization of the rearrangement goal position
    # should also be displayed
    debug_render_goal: bool = True
    robot_joint_start_noise: float = 0.0
    # Rearrange agent setup
    ctrl_freq: float = 120.0
    ac_freq_ratio: int = 4
    load_objs: bool = False
    # Rearrange agent grasping
    hold_thresh: float = 0.15
    grasp_impulse: float = 10000.0
    # we assume agent(s) to be set explicitly
    agents: Dict[str, AgentConfig] = MISSING
    # agents_order specifies the order in which the agents
    # are stored on the habitat-sim side.
    # In other words, the order to return the observations and accept
    # the actions when using the environment API.
    # If the number of agents is greater than one,
    # then agents_order has to be set explicitly.
    agents_order: List[str] = MISSING
    habitat_sim_v0: HabitatSimV0Config = HabitatSimV0Config()
    # ep_info is added to the config in some rearrange tasks inside
    # merge_sim_episode_with_object_config
    ep_info: Optional[Any] = None
    # instance ids are recorded in the panoptic sensor starting from `instance_ids_start`
    instance_ids_start: Optional[int] = 50


@attr.s(auto_attribs=True, slots=True)
class PyrobotSensor(HabitatBaseConfig):
    pass


@attr.s(auto_attribs=True, slots=True)
class PyrobotVisualSensorConfig(PyrobotSensor):
    type: str = MISSING
    height: int = 480
    width: int = 640


@attr.s(auto_attribs=True, slots=True)
class PyrobotRGBSensorConfig(PyrobotVisualSensorConfig):
    type: str = "PyRobotRGBSensor"
    center_crop: bool = False


@attr.s(auto_attribs=True, slots=True)
class PyrobotDepthSensorConfig(PyrobotVisualSensorConfig):
    type: str = "PyRobotDepthSensor"
    min_depth: float = 0.0
    max_depth: float = 5.0
    normalize_depth: bool = True
    center_crop: bool = False


@attr.s(auto_attribs=True, slots=True)
class PyrobotBumpSensorConfig(PyrobotSensor):
    type: str = "PyRobotBumpSensor"


@attr.s(auto_attribs=True, slots=True)
class LocobotConfig(HabitatBaseConfig):
    actions: List[str] = ["base_actions", "camera_actions"]
    base_actions: List[str] = ["go_to_relative", "go_to_absolute"]

    camera_actions: List[str] = ["set_pan", "set_tilt", "set_pan_tilt"]


@attr.s(auto_attribs=True, slots=True)
class PyrobotConfig(HabitatBaseConfig):
    # types of robots supported:
    robots: List[str] = ["locobot"]
    robot: str = "locobot"
    sensors: Dict[str, PyrobotSensor] = {
        "rgb_sensor": PyrobotRGBSensorConfig(),
        "depth_sensor": PyrobotDepthSensorConfig(),
        "bump_sensor": PyrobotBumpSensorConfig(),
    }
    base_controller: str = "proportional"
    base_planner: str = "none"
    locobot: LocobotConfig = LocobotConfig()


@attr.s(auto_attribs=True, slots=True)
class DatasetConfig(HabitatBaseConfig):
    type: str = "PointNav-v1"
    split: str = "train"
    scenes_dir: str = "data/scene_datasets"
    content_scenes: List[str] = ["*"]
    data_path: str = (
        "data/datasets/pointnav/"
        "habitat-test-scenes/v1/{split}/{split}.json.gz"
    )


@attr.s(auto_attribs=True, slots=True)
class OVMMDatasetConfig(DatasetConfig):
    type: str = "OVMMDataset-v0"
    viewpoints_matrix_path: Optional[
        str
    ] = "data/datasets/floorplanner/rearrange/v2/{split}/cat_rearrange_floorplanner_viewpoints_matrix.npy"
    transformations_matrix_path: Optional[
        str
    ] = "data/datasets/floorplanner/rearrange/v2/{split}/cat_rearrange_floorplanner_transformations_matrix.npy"
    # Other datasets do not allow using a subset of episodes
    episode_indices_range: Optional[Tuple[int, int]] = None
    episode_ids: List[int] = []


@attr.s(auto_attribs=True, slots=True)
class GymConfig(HabitatBaseConfig):
    auto_name: str = ""
    obs_keys: Optional[List[str]] = None
    action_keys: Optional[List[str]] = None
    achieved_goal_keys: List = []
    desired_goal_keys: List[str] = []


@attr.s(auto_attribs=True, slots=True)
class HabitatConfig(HabitatBaseConfig):
    seed: int = 100
    # GymHabitatEnv works for all Habitat tasks, including Navigation and
    # Rearrange. To use a gym environment from the registry, use the
    # GymRegistryEnv. Any other environment needs to be created and registered.
    env_task: str = "GymHabitatEnv"
    # The dependencies for launching the GymRegistryEnv environments.
    # Modules listed here will be imported prior to making the environment with
    # gym.make()
    env_task_gym_dependencies: List = []
    # The key of the gym environment in the registry to use in GymRegistryEnv
    # for example: `Cartpole-v0`
    env_task_gym_id: str = ""
    environment: EnvironmentConfig = EnvironmentConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    task: TaskConfig = MISSING
    dataset: DatasetConfig = MISSING
    gym: GymConfig = GymConfig()


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()

cs.store(group="habitat", name="habitat_config_base", node=HabitatConfig)
cs.store(
    group="habitat.environment",
    name="environment_config_schema",
    node=EnvironmentConfig,
)

cs.store(
    package="habitat.task",
    group="habitat/task",
    name="task_config_base",
    node=TaskConfig,
)

# Agent Config
cs.store(
    group="habitat/simulator/agents",
    name="agent_base",
    node=AgentConfig,
)

cs.store(
    package="habitat.task.actions.stop",
    group="habitat/task/actions",
    name="stop",
    node=StopActionConfig,
)
cs.store(
    package="habitat.task.actions.move_forward",
    group="habitat/task/actions",
    name="move_forward",
    node=MoveForwardActionConfig,
)
cs.store(
    package="habitat.task.actions.turn_left",
    group="habitat/task/actions",
    name="turn_left",
    node=TurnLeftActionConfig,
)
cs.store(
    package="habitat.task.actions.turn_right",
    group="habitat/task/actions",
    name="turn_right",
    node=TurnRightActionConfig,
)
cs.store(
    package="habitat.task.actions.look_up",
    group="habitat/task/actions",
    name="look_up",
    node=LookUpActionConfig,
)
cs.store(
    package="habitat.task.actions.look_down",
    group="habitat/task/actions",
    name="look_down",
    node=LookDownActionConfig,
)
cs.store(
    package="habitat.task.actions.arm_action",
    group="habitat/task/actions",
    name="arm_action",
    node=ArmActionConfig,
)
cs.store(
    package="habitat.task.actions.base_velocity",
    group="habitat/task/actions",
    name="base_velocity",
    node=BaseVelocityActionConfig,
)
cs.store(
    package="habitat.task.actions.empty",
    group="habitat/task/actions",
    name="empty",
    node=EmptyActionConfig,
)
cs.store(
    package="habitat.task.actions.rearrange_stop",
    group="habitat/task/actions",
    name="rearrange_stop",
    node=RearrangeStopActionConfig,
)
cs.store(
    package="habitat.task.actions.manipulation_mode",
    group="habitat/task/actions",
    name="manipulation_mode",
    node=ManipulationModeActionConfig,
)
cs.store(
    package="habitat.task.actions.answer",
    group="habitat/task/actions",
    name="answer",
    node=AnswerActionConfig,
)
cs.store(
    package="habitat.task.actions.oracle_nav_action",
    group="habitat/task/actions",
    name="oracle_nav_action",
    node=OracleNavActionConfig,
)

# Dataset Config Schema
cs.store(
    package="habitat.dataset",
    group="habitat/dataset",
    name="dataset_config_schema",
    node=DatasetConfig,
)

# Dataset Config Schema
cs.store(
    package="habitat.dataset",
    group="habitat/dataset",
    name="ovmm_dataset_config_schema",
    node=OVMMDatasetConfig,
)

# Simulator Sensors
cs.store(
    group="habitat/simulator/sim_sensors",
    name="rgb_sensor",
    node=HabitatSimRGBSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="depth_sensor",
    node=HabitatSimDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="semantic_sensor",
    node=HabitatSimSemanticSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="equirect_rgb_sensor",
    node=HabitatSimEquirectangularRGBSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="equirect_depth_sensor",
    node=HabitatSimEquirectangularDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="equirect_semantic_sensor",
    node=HabitatSimEquirectangularSemanticSensorConfig,
)


cs.store(
    group="habitat/simulator/sim_sensors",
    name="arm_depth_sensor",
    node=ArmDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="arm_rgb_sensor",
    node=ArmRGBSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="head_depth_sensor",
    node=HeadDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="head_rgb_sensor",
    node=HeadRGBSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="head_panoptic_sensor",
    node=HeadPanopticSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="third_depth_sensor",
    node=ThirdDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="third_rgb_sensor",
    node=ThirdRGBSensorConfig,
)


# Task Sensors
cs.store(
    package="habitat.task.lab_sensors.gps_sensor",
    group="habitat/task/lab_sensors",
    name="gps_sensor",
    node=GPSSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.compass_sensor",
    group="habitat/task/lab_sensors",
    name="compass_sensor",
    node=CompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.robot_start_gps_sensor",
    group="habitat/task/lab_sensors",
    name="robot_start_gps_sensor",
    node=RobotStartGPSSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.robot_start_compass_sensor",
    group="habitat/task/lab_sensors",
    name="robot_start_compass_sensor",
    node=RobotStartCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.pointgoal_with_gps_compass_sensor",
    group="habitat/task/lab_sensors",
    name="pointgoal_with_gps_compass_sensor",
    node=PointGoalWithGPSCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.objectgoal_sensor",
    group="habitat/task/lab_sensors",
    name="objectgoal_sensor",
    node=ObjectGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.imagegoal_sensor",
    group="habitat/task/lab_sensors",
    name="imagegoal_sensor",
    node=ImageGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.instance_imagegoal_sensor",
    group="habitat/task/lab_sensors",
    name="instance_imagegoal_sensor",
    node=InstanceImageGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.instance_imagegoal_hfov_sensor",
    group="habitat/task/lab_sensors",
    name="instance_imagegoal_hfov_sensor",
    node=InstanceImageGoalHFOVSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.target_start_sensor",
    group="habitat/task/lab_sensors",
    name="target_start_sensor",
    node=TargetStartSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.goal_sensor",
    group="habitat/task/lab_sensors",
    name="goal_sensor",
    node=GoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.abs_target_start_sensor",
    group="habitat/task/lab_sensors",
    name="abs_target_start_sensor",
    node=AbsTargetStartSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.abs_goal_sensor",
    group="habitat/task/lab_sensors",
    name="abs_goal_sensor",
    node=AbsGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.joint_sensor",
    group="habitat/task/lab_sensors",
    name="joint_sensor",
    node=JointSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.end_effector_sensor",
    group="habitat/task/lab_sensors",
    name="end_effector_sensor",
    node=EEPositionSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.is_holding_sensor",
    group="habitat/task/lab_sensors",
    name="is_holding_sensor",
    node=IsHoldingSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.relative_resting_pos_sensor",
    group="habitat/task/lab_sensors",
    name="relative_resting_pos_sensor",
    node=RelativeRestingPositionSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.instruction_sensor",
    group="habitat/task/lab_sensors",
    name="instruction_sensor",
    node=InstructionSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.question_sensor",
    group="habitat/task/lab_sensors",
    name="question_sensor",
    node=QuestionSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.object_sensor",
    group="habitat/task/lab_sensors",
    name="object_sensor",
    node=TargetCurrentSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.joint_velocity_sensor",
    group="habitat/task/lab_sensors",
    name="joint_velocity_sensor",
    node=JointVelocitySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.target_start_gps_compass_sensor",
    group="habitat/task/lab_sensors",
    name="target_start_gps_compass_sensor",
    node=TargetStartGpsCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.target_goal_gps_compass_sensor",
    group="habitat/task/lab_sensors",
    name="target_goal_gps_compass_sensor",
    node=TargetGoalGpsCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.nav_to_skill_sensor",
    group="habitat/task/lab_sensors",
    name="nav_to_skill_sensor",
    node=NavToSkillSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.nav_goal_sensor",
    group="habitat/task/lab_sensors",
    name="nav_goal_sensor",
    node=NavGoalPointGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.object_category_sensor",
    group="habitat/task/lab_sensors",
    name="object_category_sensor",
    node=ObjectCategorySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.goal_receptacle_sensor",
    group="habitat/task/lab_sensors",
    name="goal_receptacle_sensor",
    node=GoalReceptacleSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.start_receptacle_sensor",
    group="habitat/task/lab_sensors",
    name="start_receptacle_sensor",
    node=StartReceptacleSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.object_embedding_sensor",
    group="habitat/task/lab_sensors",
    name="object_embedding_sensor",
    node=ObjectEmbeddingSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.object_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="object_segmentation_sensor",
    node=ObjectSegmentationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.start_recep_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="start_recep_segmentation_sensor",
    node=StartRecepSegmentationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.goal_recep_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="goal_recep_segmentation_sensor",
    node=GoalRecepSegmentationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.camera_pose_sensor",
    group="habitat/task/lab_sensors",
    name="camera_pose_sensor",
    node=CameraPoseSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.receptacle_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="receptacle_segmentation_sensor",
    node=ReceptacleSegmentationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.ovmm_nav_goal_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="ovmm_nav_goal_segmentation_sensor",
    node=OVMMNavGoalSegmentationSensorConfig,
)

# Task Measurements
cs.store(
    package="habitat.task.measurements.top_down_map",
    group="habitat/task/measurements",
    name="top_down_map",
    node=TopDownMapMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.distance_to_goal",
    group="habitat/task/measurements",
    name="distance_to_goal",
    node=DistanceToGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.distance_to_goal_reward",
    group="habitat/task/measurements",
    name="distance_to_goal_reward",
    node=DistanceToGoalRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.success",
    group="habitat/task/measurements",
    name="success",
    node=SuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.spl",
    group="habitat/task/measurements",
    name="spl",
    node=SPLMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.soft_spl",
    group="habitat/task/measurements",
    name="soft_spl",
    node=SoftSPLMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.num_steps",
    group="habitat/task/measurements",
    name="num_steps",
    node=NumStepsMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.robot_force",
    group="habitat/task/measurements",
    name="robot_force",
    node=RobotForceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.force_terminate",
    group="habitat/task/measurements",
    name="force_terminate",
    node=ForceTerminateMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.robot_collisions_terminate",
    group="habitat/task/measurements",
    name="robot_collisions_terminate",
    node=RobotCollisionsTerminateMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.end_effector_to_object_distance",
    group="habitat/task/measurements",
    name="end_effector_to_object_distance",
    node=EndEffectorToObjectDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.end_effector_to_rest_distance",
    group="habitat/task/measurements",
    name="end_effector_to_rest_distance",
    node=EndEffectorToRestDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.end_effector_to_goal_distance",
    group="habitat/task/measurements",
    name="end_effector_to_goal_distance",
    node=EndEffectorToGoalDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pick_distance_to_goal",
    group="habitat/task/measurements",
    name="pick_distance_to_goal",
    node=PickDistanceToGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pick_distance_to_goal_reward",
    group="habitat/task/measurements",
    name="pick_distance_to_goal_reward",
    node=PickDistanceToGoalRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.did_pick_object",
    group="habitat/task/measurements",
    name="did_pick_object",
    node=DidPickObjectMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.did_violate_hold_constraint",
    group="habitat/task/measurements",
    name="did_violate_hold_constraint",
    node=DidViolateHoldConstraintMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pick_reward",
    group="habitat/task/measurements",
    name="pick_reward",
    node=RearrangePickRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pick_success",
    group="habitat/task/measurements",
    name="pick_success",
    node=RearrangePickSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.answer_accuracy",
    group="habitat/task/measurements",
    name="answer_accuracy",
    node=AnswerAccuracyMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_rot_dist_to_goal",
    group="habitat/task/measurements",
    name="ovmm_rot_dist_to_goal",
    node=OVMMRotDistToGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.episode_info",
    group="habitat/task/measurements",
    name="episode_info",
    node=EpisodeInfoMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.robot_colls",
    group="habitat/task/measurements",
    name="robot_colls",
    node=RobotCollisionsMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.object_to_goal_distance",
    group="habitat/task/measurements",
    name="object_to_goal_distance",
    node=ObjectToGoalDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.obj_at_goal",
    group="habitat/task/measurements",
    name="obj_at_goal",
    node=ObjAtGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.obj_anywhere_on_goal",
    group="habitat/task/measurements",
    name="obj_anywhere_on_goal",
    node=ObjAnywhereOnGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.place_success",
    group="habitat/task/measurements",
    name="place_success",
    node=PlaceSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.placement_stability",
    group="habitat/task/measurements",
    name="placement_stability",
    node=PlacementStabilityMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.place_reward",
    group="habitat/task/measurements",
    name="place_reward",
    node=PlaceRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.navmesh_collision",
    group="habitat/task/measurements",
    name="navmesh_collision",
    node=NavmeshCollisionMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_object_to_place_goal_distance",
    group="habitat/task/measurements",
    name="ovmm_object_to_place_goal_distance",
    node=OVMMObjectToPlaceGoalDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_ee_to_place_goal_distance",
    group="habitat/task/measurements",
    name="ovmm_ee_to_place_goal_distance",
    node=OVMMEEToPlaceGoalDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_place_reward",
    group="habitat/task/measurements",
    name="ovmm_place_reward",
    node=OVMMPlaceRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_placement_stability",
    group="habitat/task/measurements",
    name="ovmm_placement_stability",
    node=OVMMPlacementStabilityMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_place_success",
    group="habitat/task/measurements",
    name="ovmm_place_success",
    node=OVMMPlaceSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.object_at_rest",
    group="habitat/task/measurements",
    name="object_at_rest",
    node=ObjectAtRestMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.picked_object_linear_vel",
    group="habitat/task/measurements",
    name="picked_object_linear_vel",
    node=PickedObjectLinearVelMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.picked_object_angular_vel",
    group="habitat/task/measurements",
    name="picked_object_angular_vel",
    node=PickedObjectAngularVelMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_find_object_phase_success",
    group="habitat/task/measurements",
    name="ovmm_find_object_phase_success",
    node=OVMMFindObjectPhaseSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_pick_object_phase_success",
    group="habitat/task/measurements",
    name="ovmm_pick_object_phase_success",
    node=OVMMPickObjectPhaseSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_place_object_phase_success",
    group="habitat/task/measurements",
    name="ovmm_place_object_phase_success",
    node=OVMMPlaceObjectPhaseSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_find_recep_phase_success",
    group="habitat/task/measurements",
    name="ovmm_find_recep_phase_success",
    node=OVMMFindRecepPhaseSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.move_objects_reward",
    group="habitat/task/measurements",
    name="move_objects_reward",
    node=MoveObjectsRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.does_want_terminate",
    group="habitat/task/measurements",
    name="does_want_terminate",
    node=DoesWantTerminateMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.composite_success",
    group="habitat/task/measurements",
    name="composite_success",
    node=CompositeSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.gfx_replay_measure",
    group="habitat/task/measurements",
    name="gfx_replay_measure",
    node=GfxReplayMeasureMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.composite_stage_goals",
    group="habitat/task/measurements",
    name="composite_stage_goals",
    node=CompositeStageGoalsMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ee_dist_to_marker",
    group="habitat/task/measurements",
    name="ee_dist_to_marker",
    node=EndEffectorDistToMarkerMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.art_obj_at_desired_state",
    group="habitat/task/measurements",
    name="art_obj_at_desired_state",
    node=ArtObjAtDesiredStateMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.art_obj_state",
    group="habitat/task/measurements",
    name="art_obj_state",
    node=ArtObjStateMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.art_obj_success",
    group="habitat/task/measurements",
    name="art_obj_success",
    node=ArtObjSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.art_obj_reward",
    group="habitat/task/measurements",
    name="art_obj_reward",
    node=ArtObjRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.nav_to_pos_succ",
    group="habitat/task/measurements",
    name="nav_to_pos_succ",
    node=NavToPosSuccMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.rot_dist_to_goal",
    group="habitat/task/measurements",
    name="rot_dist_to_goal",
    node=RotDistToGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.rearrange_nav_to_obj_success",
    group="habitat/task/measurements",
    name="rearrange_nav_to_obj_success",
    node=NavToObjSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.rearrange_nav_to_obj_reward",
    group="habitat/task/measurements",
    name="rearrange_nav_to_obj_reward",
    node=NavToObjRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.bad_called_terminate",
    group="habitat/task/measurements",
    name="bad_called_terminate",
    node=BadCalledTerminateMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.dist_to_goal",
    group="habitat/task/measurements",
    name="dist_to_goal",
    node=DistToGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_nav_to_obj_reward",
    group="habitat/task/measurements",
    name="ovmm_nav_to_obj_reward",
    node=OVMMNavToObjRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.dist_to_pick_goal",
    group="habitat/task/measurements",
    name="ovmm_dist_to_pick_goal",
    node=OVMMDistToPickGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.dist_to_place_goal",
    group="habitat/task/measurements",
    name="ovmm_dist_to_place_goal",
    node=OVMMDistToPlaceGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.rot_dist_to_pick_goal",
    group="habitat/task/measurements",
    name="ovmm_rot_dist_to_pick_goal",
    node=OVMMRotDistToPickGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.rot_dist_to_place_goal",
    group="habitat/task/measurements",
    name="ovmm_rot_dist_to_place_goal",
    node=OVMMRotDistToPlaceGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.nav_to_pick_succ",
    group="habitat/task/measurements",
    name="ovmm_nav_to_pick_succ",
    node=OVMMNavToPickSuccMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.nav_to_place_succ",
    group="habitat/task/measurements",
    name="ovmm_nav_to_place_succ",
    node=OVMMNavToPlaceSuccMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.ovmm_nav_to_obj_success",
    group="habitat/task/measurements",
    name="ovmm_nav_to_obj_success",
    node=OVMMNavToObjSuccMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.nav_orient_to_pick_succ",
    group="habitat/task/measurements",
    name="ovmm_nav_orient_to_pick_succ",
    node=OVMMNavOrientToPickSuccMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.nav_orient_to_place_succ",
    group="habitat/task/measurements",
    name="ovmm_nav_orient_to_place_succ",
    node=OVMMNavOrientToPlaceSuccMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pick_goal_iou_coverage",
    group="habitat/task/measurements",
    name="pick_goal_iou_coverage",
    node=PickGoalIoUCoverageMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.place_goal_iou_coverage",
    group="habitat/task/measurements",
    name="place_goal_iou_coverage",
    node=PlaceGoalIoUCoverageMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.target_iou_coverage",
    group="habitat/task/measurements",
    name="target_iou_coverage",
    node=TargetIoUCoverageMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.rearrange_reach_reward",
    group="habitat/task/measurements",
    name="rearrange_reach_reward",
    node=RearrangeReachRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.rearrange_reach_success",
    group="habitat/task/measurements",
    name="rearrange_reach_success",
    node=RearrangeReachSuccessMeasurementConfig,
)


from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://habitat/config/",
        )


def register_hydra_plugin(plugin) -> None:
    """Hydra users should call this function before invoking @hydra.main"""
    Plugins.instance().register(plugin)
