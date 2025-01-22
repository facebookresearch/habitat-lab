#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

# __all__ is used for documentation. Only put in this list the configurations
# that have proper documentation for.
__all__ = [
    # HABITAT
    "HabitatConfig",
    # DATASET
    "DatasetConfig",
    # TASK
    "TaskConfig",
    # ENVIRONMENT
    "EnvironmentConfig",
    # NAVIGATION ACTIONS
    "StopActionConfig",
    "MoveForwardActionConfig",
    "TurnLeftActionConfig",
    "TurnLeftActionConfig",
    "TurnRightActionConfig",
    "LookUpActionConfig",
    "LookDownActionConfig",
    # NAVIGATION MEASURES
    "NumStepsMeasurementConfig",
    "DistanceToGoalMeasurementConfig",
    "SuccessMeasurementConfig",
    "SPLMeasurementConfig",
    "SoftSPLMeasurementConfig",
    "DistanceToGoalRewardMeasurementConfig",
    # NAVIGATION LAB SENSORS
    "ObjectGoalSensorConfig",
    "InstanceImageGoalSensorConfig",
    "InstanceImageGoalHFOVSensorConfig",
    "CompassSensorConfig",
    "GPSSensorConfig",
    "PointGoalWithGPSCompassSensorConfig",
    "HumanoidDetectorSensorConfig",
    "ArmDepthBBoxSensorConfig",
    "SpotHeadStereoDepthSensorConfig",
    # REARRANGEMENT ACTIONS
    "EmptyActionConfig",
    "ArmActionConfig",
    "BaseVelocityActionConfig",
    "HumanoidJointActionConfig",
    "HumanoidPickActionConfig",
    "RearrangeStopActionConfig",
    "OracleNavActionConfig",
    "SelectBaseOrArmActionConfig",
    # REARRANGEMENT LAB SENSORS
    "RelativeRestingPositionSensorConfig",
    "IsHoldingSensorConfig",
    "EEPositionSensorConfig",
    "JointSensorConfig",
    "HumanoidJointSensorConfig",
    "TargetStartSensorConfig",
    "GoalSensorConfig",
    "TargetStartGpsCompassSensorConfig",
    "InitialGpsCompassSensorConfig",
    "TargetGoalGpsCompassSensorConfig",
    # REARRANGEMENT MEASUREMENTS
    "EndEffectorToRestDistanceMeasurementConfig",
    "RobotForceMeasurementConfig",
    "DoesWantTerminateMeasurementConfig",
    "ForceTerminateMeasurementConfig",
    "ObjectToGoalDistanceMeasurementConfig",
    "ObjAtGoalMeasurementConfig",
    "ArtObjAtDesiredStateMeasurementConfig",
    "RotDistToGoalMeasurementConfig",
    "PddlStageGoalsMeasurementConfig",
    "NavToPosSuccMeasurementConfig",
    "SocialNavStatsMeasurementConfig",
    "NavSeekSuccessMeasurementConfig",
    # REARRANGEMENT MEASUREMENTS TASK REWARDS AND MEASURES
    "RearrangePickSuccessMeasurementConfig",
    "RearrangePickRewardMeasurementConfig",
    "PlaceSuccessMeasurementConfig",
    "PlaceRewardMeasurementConfig",
    "ArtObjSuccessMeasurementConfig",
    "ArtObjRewardMeasurementConfig",
    "NavToObjSuccessMeasurementConfig",
    "NavToObjRewardMeasurementConfig",
    "PddlSuccessMeasurementConfig",
    # PROFILING MEASURES
    "RuntimePerfStatsMeasurementConfig",
]


@dataclass
class HabitatBaseConfig:
    pass


@dataclass
class IteratorOptionsConfig(HabitatBaseConfig):
    cycle: bool = True
    shuffle: bool = True
    group_by_scene: bool = True
    num_episode_sample: int = -1
    max_scene_repeat_episodes: int = -1
    max_scene_repeat_steps: int = int(1e4)
    step_repetition_range: float = 0.2


@dataclass
class EnvironmentConfig(HabitatBaseConfig):
    r"""
    Some habitat environment configurations.

    :property max_episode_steps: The maximum number of environment steps before the episode ends.
    :property max_episode_seconds: The maximum number of wall-clock seconds before the episode ends.
    """
    max_episode_steps: int = 1000
    max_episode_seconds: int = 10000000
    iterator_options: IteratorOptionsConfig = IteratorOptionsConfig()


# -----------------------------------------------------------------------------
# # Actions
# -----------------------------------------------------------------------------
@dataclass
class ActionConfig(HabitatBaseConfig):
    type: str = MISSING


@dataclass
class StopActionConfig(ActionConfig):
    r"""
    In Navigation tasks only, the stop action is a discrete action. When called, the Agent
    will request to stop the navigation task. Note that this action is needed to
    succeed in a Navigation task since the Success is determined by the Agent calling
    the stop action within range of the target.
    Note that this is different from the RearrangeStopActionConfig that works for
    Rearrangement tasks only instead of the Navigation tasks.
    """
    type: str = "StopAction"


@dataclass
class EmptyActionConfig(ActionConfig):
    r"""
    In Navigation tasks only, the pass action. The robot will do nothing.
    """
    type: str = "EmptyAction"


# -----------------------------------------------------------------------------
# # NAVIGATION actions
# -----------------------------------------------------------------------------


@dataclass
class DiscreteNavigationActionConfig(ActionConfig):
    tilt_angle: int = 15  # angle to tilt the camera up or down in degrees


@dataclass
class MoveForwardActionConfig(DiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will move the robot forward by
    a fixed amount determined by the SimulatorConfig.forward_step_size amount.
    """
    type: str = "MoveForwardAction"


@dataclass
class TurnLeftActionConfig(DiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot to the left
    by a fixed amount determined by the SimulatorConfig.turn_angle amount.
    """
    type: str = "TurnLeftAction"


@dataclass
class TurnRightActionConfig(DiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot to the right
    by a fixed amount determined by the SimulatorConfig.turn_angle amount.
    """
    type: str = "TurnRightAction"


@dataclass
class LookUpActionConfig(DiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera up
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "LookUpAction"


@dataclass
class LookDownActionConfig(DiscreteNavigationActionConfig):
    r"""
    In Navigation tasks only, this discrete action will rotate the robot's camera down
    by a fixed amount determined by the SimulatorConfig.tilt_angle amount.
    """
    type: str = "LookDownAction"


@dataclass
class TeleportActionConfig(ActionConfig):
    type: str = "TeleportAction"


@dataclass
class VelocityControlActionConfig(ActionConfig):
    type: str = "VelocityAction"
    # meters/sec
    lin_vel_range: List[float] = field(default_factory=lambda: [0.0, 0.25])
    # deg/sec
    ang_vel_range: List[float] = field(default_factory=lambda: [-10.0, 10.0])
    min_abs_lin_speed: float = 0.025  # meters/sec
    min_abs_ang_speed: float = 1.0  # # deg/sec
    time_step: float = 1.0  # seconds


# -----------------------------------------------------------------------------
# # REARRANGE actions
# -----------------------------------------------------------------------------
@dataclass
class ArmActionConfig(ActionConfig):
    r"""
    In Rearrangement tasks only, the action that will move the robot arm around. The action represents to delta angle (in radians) of each joint.

    :property grasp_thresh_dist: The grasp action will only work on the closest object if its distance to the end effector is smaller than this value. Only for `MagicGraspAction` grip_controller.
    :property grip_controller: Can either be None,  `MagicGraspAction` or `SuctionGraspAction`. If None, the arm will be unable to grip object. Magic grasp will grasp the object if the end effector is within grasp_thresh_dist of an object, with `SuctionGraspAction`, the object needs to be in contact with the end effector.
    :property arm_joint_limit: If the user specify the joint limits. The dimension needs to be the same as arm_joint_dimensionality. Format: [[min_limit, max_limit]]
    :property gaze_distance_range: The gaze action will only work on the closet object if its distance to the end effector is smaller than this value. Only for `GazeGraspAction` grip_controller.
    :property center_cone_angle_threshold: The threshold angle between the line of sight and center_cone_vector. Only for `GazeGraspAction` grip_controller.
    :property center_cone_vector: The vector that the camera's line of sight should be when grasping the object. Only for `GazeGraspAction` grip_controller.
    """
    type: str = "ArmAction"
    arm_controller: str = "ArmRelPosAction"
    grip_controller: Optional[str] = None
    arm_joint_mask: Optional[List[int]] = None
    arm_joint_dimensionality: int = 7
    arm_joint_limit: Optional[List[float]] = None
    grasp_thresh_dist: float = 0.15
    disable_grip: bool = False
    delta_pos_limit: float = 0.0125
    ee_ctrl_lim: float = 0.015
    should_clip: bool = False
    render_ee_target: bool = False
    gaze_distance_range: Optional[List[float]] = None
    center_cone_angle_threshold: float = 0.0
    center_cone_vector: Optional[List[float]] = None
    auto_grasp: bool = False


@dataclass
class BaseVelocityActionConfig(ActionConfig):
    r"""
    In Rearrangement only. Corresponds to the base velocity. Contains two continuous actions, the first one controls forward and backward motion, the second the rotation.
    """
    type: str = "BaseVelAction"
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    allow_back: bool = True


@dataclass
class BaseVelocityNonCylinderActionConfig(ActionConfig):
    r"""
    In Rearrangement only for the non cylinder shape of the robot. Corresponds to the base velocity. Contains two continuous actions, the first one controls forward and backward motion, the second the rotation.
    """
    type: str = "BaseVelNonCylinderAction"
    # The max longitudinal and lateral linear speeds of the robot
    lin_speed: float = 10.0
    longitudinal_lin_speed: float = 10.0
    lateral_lin_speed: float = 10.0
    # The max angular speed of the robot
    ang_speed: float = 10.0
    # If we want to do sliding or not
    allow_dyn_slide: bool = False
    # If we allow the robot to move back or not
    allow_back: bool = True
    # There is a collision if the difference between the clamped NavMesh position and target position
    # is more than collision_threshold for any point.
    collision_threshold: float = 1e-5
    # If we allow the robot to move laterally.
    enable_lateral_move: bool = False
    # If the condition of sliding includes the checking of rotation
    enable_rotation_check_for_dyn_slide: bool = True


@dataclass
class HumanoidJointActionConfig(ActionConfig):
    r"""
    In Rearrangement only. Corresponds to actions to change the humanoid joints. Contains the parameter num_joints, indicating the joints that can be modified.
    """
    type: str = "HumanoidJointAction"
    num_joints: int = 54


@dataclass
class HumanoidPickActionConfig(ActionConfig):
    r"""
    In rearrangement tasks only. Config for humanoid to reach objects using IK. For now only contains the number of joints. May be extended with duration of action
    """
    type: str = "HumanoidPickAction"
    # Number of joints in the humanoid body, 54 for SMPL-X, 17 for SMPL
    num_joints: int = 54
    # The amount we should move on every call to humanoid pick action
    dist_move_per_step: float = 0.04
    # The distance at which we will snap/desnap an object, and start retracting the hand
    dist_to_snap: float = 0.02


@dataclass
class RearrangeStopActionConfig(ActionConfig):
    r"""
    In rearrangement tasks only, if the robot calls this action, the task will end.
    """
    type: str = "RearrangeStopAction"


@dataclass
class PddlApplyActionConfig(ActionConfig):
    type: str = "PddlApplyAction"


@dataclass
class OracleNavActionConfig(ActionConfig):
    """
    Rearrangement Only, Oracle navigation action.
    This action takes as input a discrete ID which refers to an object in the
    PDDL domain. The oracle navigation controller then computes the actions to
    navigate to that desired object.
    """

    type: str = "OracleNavAction"
    # Whether the motion is in the form of base_velocity or human_joints
    motion_control: str = "base_velocity"
    num_joints: int = 17
    turn_velocity: float = 1.0
    forward_velocity: float = 1.0
    turn_thresh: float = 0.1
    dist_thresh: float = 0.2
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    allow_back: bool = True
    spawn_max_dist_to_obj: float = 2.0
    num_spawn_attempts: int = 200
    # For social nav training only. It controls the distance threshold
    # between the robot and the human and decide if the human wants to walk or not
    human_stop_and_walk_to_robot_distance_threshold: float = -1.0


@dataclass
class SelectBaseOrArmActionConfig(ActionConfig):
    r"""
    In rearrangement tasks only, if the robot calls this action, the task will end.
    """
    type: str = "SelectBaseOrArmAction"


# -----------------------------------------------------------------------------
# # EQA actions
# -----------------------------------------------------------------------------
@dataclass
class AnswerActionConfig(ActionConfig):
    type: str = "AnswerAction"


# -----------------------------------------------------------------------------
# # TASK_SENSORS
# -----------------------------------------------------------------------------
@dataclass
class LabSensorConfig(HabitatBaseConfig):
    type: str = MISSING


@dataclass
class PointGoalSensorConfig(LabSensorConfig):
    type: str = "PointGoalSensor"
    goal_format: str = "POLAR"
    dimensionality: int = 2


@dataclass
class PointGoalWithGPSCompassSensorConfig(PointGoalSensorConfig):
    """
    Indicates the position of the point goal in the frame of reference of the robot.
    """

    type: str = "PointGoalWithGPSCompassSensor"


@dataclass
class HumanoidDetectorSensorConfig(LabSensorConfig):
    r"""
    Check if the human is in frame
    """
    type: str = "HumanoidDetectorSensor"
    # The default human id is 100
    human_id: int = 100
    # How many pixels needed to consider that human is in frame
    human_pixel_threshold: int = 1000
    # Image based or binary based
    return_image: bool = False
    # Is the return image bounding box or not
    is_return_image_bbox: bool = False


@dataclass
class ArmDepthBBoxSensorConfig(LabSensorConfig):
    r"""
    Bounding box sensor to check if the object is in frame
    """
    type: str = "ArmDepthBBoxSensor"
    height: int = 480
    width: int = 640


@dataclass
class SpotHeadStereoDepthSensorConfig(LabSensorConfig):
    r"""
    For Spot only. Sensor fusion for inputs of Spot stereo pair depth sensor
    """
    type: str = "SpotHeadStereoDepthSensor"
    height: int = 240
    width: int = 228


@dataclass
class ObjectGoalSensorConfig(LabSensorConfig):
    r"""
    For Object Navigation tasks only. Generates a discrete observation containing
    the id of the goal object for the episode.

    :property goal_spec: A string that can take the value TASK_CATEGORY_ID or OBJECT_ID. If the value is TASK_CATEGORY_ID, then the observation will be the id of the `episode.object_category` attribute, if the value is OBJECT_ID, then the observation will be the id of the first goal object.
    :property goal_spec_max_val: If the `goal_spec` is OBJECT_ID, then `goal_spec_max_val` is the total number of different objects that can be goals. Note that this value must be greater than the largest episode goal category id.
    """
    type: str = "ObjectGoalSensor"
    goal_spec: str = "TASK_CATEGORY_ID"
    goal_spec_max_val: int = 50


@dataclass
class ImageGoalSensorConfig(LabSensorConfig):
    type: str = "ImageGoalSensor"


@dataclass
class InstanceImageGoalSensorConfig(LabSensorConfig):
    r"""
    Used only by the InstanceImageGoal Navigation task. The observation is a rendered
    image of the goal object within the scene.
    """
    type: str = "InstanceImageGoalSensor"


@dataclass
class InstanceImageGoalHFOVSensorConfig(LabSensorConfig):
    r"""
    Used only by the InstanceImageGoal Navigation task. The observation is a single
    float value corresponding to the Horizontal field of view (HFOV) in degrees of
    the image provided by the `InstanceImageGoalSensor`.
    """
    type: str = "InstanceImageGoalHFOVSensor"


@dataclass
class HeadingSensorConfig(LabSensorConfig):
    type: str = "HeadingSensor"


@dataclass
class CompassSensorConfig(LabSensorConfig):
    r"""
    For Navigation tasks only. The observation of the
    `EpisodicCompassSensor` is a single float value corresponding to
    the angle difference in radians between the current rotation of the robot and the
    start rotation of the robot along the vertical axis.
    """
    type: str = "CompassSensor"


@dataclass
class GPSSensorConfig(LabSensorConfig):
    r"""
    For Navigation tasks only. The observation of the EpisodicGPSSensor are two float values
    corresponding to the vector difference in the horizontal plane between the current position
    and the start position of the robot (in meters).
    """
    type: str = "GPSSensor"
    dimensionality: int = 2


@dataclass
class ProximitySensorConfig(LabSensorConfig):
    type: str = "ProximitySensor"
    max_detection_radius: float = 2.0


@dataclass
class JointSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Returns the joint positions of the robot.
    """
    type: str = "JointSensor"
    dimensionality: int = 7
    arm_joint_mask: Optional[List[int]] = None


@dataclass
class HumanoidJointSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Returns the joint positions of the robot.
    """
    type: str = "HumanoidJointSensor"
    dimensionality: int = 17 * 4


@dataclass
class EEPositionSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. the cartesian coordinates (3 floats) of the arm's end effector in the frame of reference of the robot's base.
    """
    type: str = "EEPositionSensor"


@dataclass
class IsHoldingSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. A single float sensor with value 1.0 if the robot is grasping any object and 0.0 otherwise.
    """
    type: str = "IsHoldingSensor"


@dataclass
class RelativeRestingPositionSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Sensor for the relative position of the end-effector's resting position, relative to the end-effector's current position.
    The three values correspond to the cartesian coordinates of the resting position in the frame of reference of the end effector.
    The desired resting position is determined by the habitat.task.desired_resting_position coordinates relative to the robot's base.
    """

    type: str = "RelativeRestingPositionSensor"


@dataclass
class JointVelocitySensorConfig(LabSensorConfig):
    type: str = "JointVelocitySensor"
    dimensionality: int = 7


@dataclass
class OracleNavigationActionSensorConfig(LabSensorConfig):
    type: str = "OracleNavigationActionSensor"


@dataclass
class RestingPositionSensorConfig(LabSensorConfig):
    type: str = "RestingPositionSensor"


@dataclass
class ArtJointSensorConfig(LabSensorConfig):
    type: str = "ArtJointSensor"


@dataclass
class NavGoalSensorConfig(LabSensorConfig):
    type: str = "NavGoalSensor"


@dataclass
class ArtJointSensorNoVelSensorConfig(LabSensorConfig):
    type: str = "ArtJointSensorNoVel"  # TODO: add "Sensor" suffix


@dataclass
class MarkerRelPosSensorConfig(LabSensorConfig):
    type: str = "MarkerRelPosSensor"


@dataclass
class TargetStartSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Returns the relative position from end effector to a target object that needs to be picked up.
    """
    type: str = "TargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class TargetCurrentSensorConfig(LabSensorConfig):
    type: str = "TargetCurrentSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class GoalSensorConfig(LabSensorConfig):
    """
    Rearrangement only. Returns the relative position from end effector to a goal position in which the agent needs to place an object.
    """

    type: str = "GoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class NavGoalPointGoalSensorConfig(LabSensorConfig):
    type: str = "NavGoalPointGoalSensor"
    goal_is_human: bool = False
    human_agent_idx: int = 1


@dataclass
class GlobalPredicatesSensorConfig(LabSensorConfig):
    type: str = "GlobalPredicatesSensor"


@dataclass
class MultiAgentGlobalPredicatesSensorConfig(LabSensorConfig):
    type: str = "MultiAgentGlobalPredicatesSensor"


@dataclass
class AreAgentsWithinThresholdConfig(LabSensorConfig):
    type: str = "AreAgentsWithinThreshold"
    x_len: Optional[float] = None
    y_len: Optional[float] = None
    agent_idx: int = 0


@dataclass
class HasFinishedOracleNavSensorConfig(LabSensorConfig):
    type: str = "HasFinishedOracleNavSensor"


@dataclass
class HasFinishedHumanoidPickSensorConfig(LabSensorConfig):
    type: str = "HasFinishedHumanoidPickSensor"


@dataclass
class OtherAgentGpsConfig(LabSensorConfig):
    type: str = "OtherAgentGps"


@dataclass
class TargetStartGpsCompassSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Returns the initial position of every object that needs to be rearranged in composite tasks, in 2D polar coordinates.
    """
    type: str = "TargetStartGpsCompassSensor"


@dataclass
class InitialGpsCompassSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Returns the relative distance to the initial starting location of the agent in 2D polar coordinates.
    """
    type: str = "InitialGpsCompassSensor"


@dataclass
class TargetGoalGpsCompassSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Returns the desired goal position of every object that needs to be rearranged in composite tasks, in 2D polar coordinates.
    """
    type: str = "TargetGoalGpsCompassSensor"


@dataclass
class NavToSkillSensorConfig(LabSensorConfig):
    type: str = "NavToSkillSensor"
    num_skills: int = 8


@dataclass
class AbsTargetStartSensorConfig(LabSensorConfig):
    type: str = "AbsTargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class AbsGoalSensorConfig(LabSensorConfig):
    type: str = "AbsGoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class DistToNavGoalSensorConfig(LabSensorConfig):
    type: str = "DistToNavGoalSensor"


@dataclass
class LocalizationSensorConfig(LabSensorConfig):
    type: str = "LocalizationSensor"


@dataclass
class QuestionSensorConfig(LabSensorConfig):
    type: str = "QuestionSensor"


@dataclass
class InstructionSensorConfig(LabSensorConfig):
    type: str = "InstructionSensor"
    instruction_sensor_uuid: str = "instruction"


# -----------------------------------------------------------------------------
# Measurements
# -----------------------------------------------------------------------------
@dataclass
class MeasurementConfig(HabitatBaseConfig):
    type: str = MISSING


@dataclass
class SuccessMeasurementConfig(MeasurementConfig):
    r"""
    For Navigation tasks only, Measures 1.0 if the robot reached a success and 0 otherwise.
    A success is defined as calling the `StopAction` when the `DistanceToGoal`
    Measure is smaller than `success_distance`.

    :property success_distance: The minimal distance the robot must be to the goal for a success.
    """
    type: str = "Success"
    success_distance: float = 0.2


@dataclass
class SPLMeasurementConfig(MeasurementConfig):
    r"""
    For Navigation tasks only, Measures the SPL (Success weighted by Path Length)
    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    Measure is always 0 except at success where it will be
    the ratio of the optimal distance from start to goal over the total distance
    traveled by the agent. Maximum value is 1.
    SPL = success * optimal_distance_to_goal / distance_traveled_so_far
    """
    type: str = "SPL"


@dataclass
class SoftSPLMeasurementConfig(MeasurementConfig):
    r"""
    For Navigation tasks only, Similar to SPL, but instead of a boolean,
    success is now calculated as 1 - (ratio of distance covered to target).
    SoftSPL = max(0, 1 - distance_to_goal / optimal_distance_to_goal) * optimal_distance_to_goal / distance_traveled_so_far
    """
    type: str = "SoftSPL"


@dataclass
class FogOfWarConfig:
    draw: bool = True
    visibility_dist: float = 5.0
    fov: int = 90


@dataclass
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


@dataclass
class CollisionsMeasurementConfig(MeasurementConfig):
    type: str = "Collisions"


@dataclass
class RuntimePerfStatsMeasurementConfig(MeasurementConfig):
    """
    If added to the measurements, this will time various sections of code in
    the simulator and task logic. If using with a multi-environment trainer
    (like DD-PPO) it is recommended to only log this stat for one environment
    since this metric can include many numbers.
    """

    type: str = "RuntimePerfStats"
    disable_logging: bool = False


@dataclass
class RobotForceMeasurementConfig(MeasurementConfig):
    r"""
    The amount of force in newton's applied by the robot. It computes both the instant and accumulated.
    """
    type: str = "RobotForce"
    min_force: float = 20.0


@dataclass
class ForceTerminateMeasurementConfig(MeasurementConfig):
    r"""
    If the force is greater than a certain threshold, this measure will be 1.0 and 0.0 otherwise.
    Note that if the measure is 1.0, the task will end as a result.

    :property max_accum_force: The threshold for the accumulated force. -1 is no threshold.
    :property max_instant_force: The threshold for the current, instant force. -1 is no threshold.
    """
    type: str = "ForceTerminate"
    max_accum_force: float = -1.0
    max_instant_force: float = -1.0


@dataclass
class RobotCollisionsMeasurementConfig(MeasurementConfig):
    type: str = "RobotCollisions"


@dataclass
class ObjectToGoalDistanceMeasurementConfig(MeasurementConfig):
    r"""
    In rearrangement only. The distance between the target object and the goal position for the object.
    """
    type: str = "ObjectToGoalDistance"


@dataclass
class EndEffectorToObjectDistanceMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorToObjectDistance"
    # Normally, we compute the L2 distance between the gripper and the object. However,
    # sometimes we also want to make sure that the gripper is facing the object with a right angle/orientation.
    # For instance, the gripper could face vertically on top of the object.
    if_consider_gaze_angle: bool = False
    # The vector that the camera's line of sight should be when grasping the object.
    center_cone_vector: Optional[List[float]] = None
    # Normally, you want the L2 distance between the gripper and the object as small as possible.
    # However, there are cases where you want to constrain the distance to be close to a specific value.
    desire_distance_between_gripper_object: float = 0.0


@dataclass
class BaseToObjectDistanceMeasurementConfig(MeasurementConfig):
    """L2 distance between the base and the object"""

    type: str = "BaseToObjectDistance"


@dataclass
class EndEffectorToRestDistanceMeasurementConfig(MeasurementConfig):
    """
    Rearrangement only. Distance between current end effector position
    and the resting position of the end effector. Requires that the
    RelativeRestingPositionSensor is attached to the agent.
    """

    type: str = "EndEffectorToRestDistance"


@dataclass
class EndEffectorToGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorToGoalDistance"


@dataclass
class ArtObjAtDesiredStateMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement open/close container tasks only. Whether the articulated object (fridge or cabinet door) towards a desired state (open or closed) as defined by the task.
    """
    type: str = "ArtObjAtDesiredState"
    use_absolute_distance: bool = True
    success_dist_threshold: float = 0.05


@dataclass
class GfxReplayMeasureMeasurementConfig(MeasurementConfig):
    type: str = "GfxReplayMeasure"


@dataclass
class EndEffectorDistToMarkerMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorDistToMarker"


@dataclass
class ArtObjStateMeasurementConfig(MeasurementConfig):
    type: str = "ArtObjState"


@dataclass
class ArtObjSuccessMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement open/close container tasks only. Requires art_obj_at_desired_state. Is 1.0 if the articulated object is in desired state and the end effector is within rest_dist_threshold of the resting position. If must_call_stop is True, the robot must also call the rearrange_stop action.
    """
    type: str = "ArtObjSuccess"
    rest_dist_threshold: float = 0.15
    must_call_stop: bool = True


@dataclass
class ArtObjRewardMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement open/close container tasks only. Requires art_obj_at_desired_state.

    :property dist_reward: At each step, the measure adds dist_reward times the distance the end effector moved towards the articulated object.
    :property wrong_grasp_end: If true, the task will end if the robot picks the wrong articulated object.
    :property wrong_grasp_pen: The penalty for picking the wrong articulated object.
    :property art_dist_reward: At each step, increments the reward by the amount the articulated moved in the correct direction.
    :property art_at_desired_state_reward: The reward for putting the articulated object in the right state.
    :property grasp_reward: The reward for grasping the correct articulated object.
    """
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
    force_pen: float = 0.0
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0
    count_coll_pen: float = -1.0
    max_count_colls: int = -1
    count_coll_end_pen: float = 1.0


@dataclass
class RotDistToGoalMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Navigation task only. The angle between the forward direction of the agent and the direction to the goal location.
    """
    type: str = "RotDistToGoal"


@dataclass
class DistToGoalMeasurementConfig(MeasurementConfig):
    type: str = "DistToGoal"


@dataclass
class BadCalledTerminateMeasurementConfig(MeasurementConfig):
    type: str = "BadCalledTerminate"
    bad_term_pen: float = 0.0
    decay_bad_term: bool = False


@dataclass
class NavToPosSuccMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Navigation task only. The value is 1.0 if the robot is within success_distance of the goal position.
    """
    type: str = "NavToPosSucc"
    success_distance: float = 1.5


@dataclass
class SocialNavStatsMeasurementConfig(MeasurementConfig):
    r"""
    Social nav stats computation
    """
    type: str = "SocialNavStats"
    # Check if the human is inside the frame or not
    check_human_in_frame: bool = False
    # The safety distance
    min_dis_human: float = 1.0
    max_dis_human: float = 2.0
    # The human id
    human_id: int = 100
    # The pixel needed
    human_detect_pixel_threshold: int = 1000
    # The total number of steps
    total_steps: int = 1500
    # If we want to compute the shortest path to human
    enable_shortest_path_computation: bool = False
    # The min distance for considering backup and yield motions
    dis_threshold_for_backup_yield: float = 1.5
    # The min vel for considering yield motion
    min_abs_vel_for_yield: float = 1.0
    # The dot product value for considering that the robot is facing human
    robot_face_human_threshold: float = 0.5
    # Agent ids
    robot_idx: int = 0
    human_idx: int = 1


@dataclass
class NavSeekSuccessMeasurementConfig(MeasurementConfig):
    r"""
    Social nav seek success measurement
    """
    type: str = "SocialNavSeekSuccess"
    # If the robot needs to look at the target
    must_look_at_targ: bool = True
    must_call_stop: bool = True
    # distance in radians.
    success_angle_dist: float = 0.261799
    # distance
    following_step_succ_threshold: int = 800
    safe_dis_min: float = 1.0
    safe_dis_max: float = 2.0
    need_to_face_human: bool = False
    use_geo_distance: bool = False
    facing_threshold: float = 0.5
    # Set the agent ids
    robot_idx: int = 0
    human_idx: int = 1


@dataclass
class NavToObjRewardMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Navigation task only. The reward for rearrangement navigation.

    :property dist_reward: At each step, the measure adds dist_reward times the distance the robot's base moved towards the goal position.
    :property should_reward_turn: If true, the robot receives a reward for turning towards the target position.
    :property angle_dist_reward: The reward multiplier for the robot turning towards the goal position.
    """
    type: str = "NavToObjReward"
    # reward the agent for facing the object?
    should_reward_turn: bool = True
    # what distance do we start giving the reward for facing the object?
    turn_reward_dist: float = 3.0
    # multiplier on the angle distance to the goal.
    angle_dist_reward: float = 1.0
    dist_reward: float = 1.0
    constraint_violate_pen: float = 1.0
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    count_coll_pen: float = -1.0
    max_count_colls: int = -1
    count_coll_end_pen: float = 1.0


@dataclass
class NavToObjSuccessMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Navigation only. Takes the value 1.0 when the Robot successfully navigated to the target object. Depends on nav_to_pos_succ.

    :property must_look_at_targ: If true, the robot must be facing the correct object in addition to being close to it.
    :property must_call_stop: If true, the robot must in addition, call the rearrange_stop action for this measure to be a success.
    :property success_angle_dist: When the robot must look at the target, this is the maximum angle in radians the robot can have when facing the object.
    """
    type: str = "NavToObjSuccess"
    must_look_at_targ: bool = True
    must_call_stop: bool = True
    # distance in radians.
    success_angle_dist: float = 0.261799


@dataclass
class RearrangeReachRewardMeasurementConfig(MeasurementConfig):
    type: str = "RearrangeReachReward"
    scale: float = 1.0
    diff_reward: bool = True
    sparse_reward: bool = False


@dataclass
class RearrangeReachSuccessMeasurementConfig(MeasurementConfig):
    type: str = "RearrangeReachSuccess"
    succ_thresh: float = 0.2


@dataclass
class NumStepsMeasurementConfig(MeasurementConfig):
    r"""
    In both Navigation and Rearrangement tasks, counts the number of steps since
    the start of the episode.
    """
    type: str = "NumStepsMeasure"


@dataclass
class ZeroMeasurementConfig(MeasurementConfig):
    r"""
    Always returns 0. Can we used for a sparse reward or a task without any
    success criteria.
    """
    type: str = "ZeroMeasure"


@dataclass
class DidPickObjectMeasurementConfig(MeasurementConfig):
    type: str = "DidPickObjectMeasure"


@dataclass
class DidViolateHoldConstraintMeasurementConfig(MeasurementConfig):
    type: str = "DidViolateHoldConstraintMeasure"


@dataclass
class MoveObjectsRewardMeasurementConfig(MeasurementConfig):
    type: str = "MoveObjectsReward"
    pick_reward: float = 1.0
    success_dist: float = 0.15
    single_rearrange_reward: float = 1.0
    dist_reward: float = 1.0
    constraint_violate_pen: float = 10.0
    force_pen: float = 0.001
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0
    count_coll_pen: float = -1.0
    max_count_colls: int = -1
    count_coll_end_pen: float = 1.0


@dataclass
class RearrangePickRewardMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Only. Requires the end_effector_sensor lab sensor. The reward for the pick task.

    :property dist_reward: At each step, the measure adds dist_reward times the distance the end effector moved towards the target.
    :property pick_reward: If the robot picks the target object, it receives pick_reward reward.
    :property drop_pen: The penalty for dropping the object.
    :property wrong_pick_pen: The penalty for picking the wrong object.
    :property force_pen: At each step, adds a penalty of force_pen times the current force on the robot.
    :property drop_obj_should_end: If true, the task will end if the robot drops the object.
    :property wrong_pick_should_end: If true, the task will end if the robot picks the wrong object.
    :property max_target_distance: default: -1. If it is positive, then we terminate the episode if the robot distance to object is above this value.
    :property max_target_distance_pen: If the robot is too far away, then we terminate the episode by giving the penality.
    :property non_desire_ee_local_pos_dis: default: -1. If positive, we terminate the episode if the robot moves the arm below this threshold
    :property non_desire_ee_local_pos_pen: If the robot moves the arm there, then we terminate the episode by giving the penality.
    :property non_desire_ee_local_pos: If given, we do not want the robot to move the arm there
    :property camera_looking_down_angle: default: -1. If positive, we check the robot camera looking angle to the ground
    :property camera_looking_down_pen: If the robot camera looking angle is too small (the robot looks down), we terminate the episode and with this much penality
    """
    type: str = "RearrangePickReward"
    dist_reward: float = 2.0
    pick_reward: float = 2.0
    constraint_violate_pen: float = 1.0
    drop_pen: float = 0.5
    wrong_pick_pen: float = 0.5
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    use_diff: bool = True
    drop_obj_should_end: bool = True
    wrong_pick_should_end: bool = True
    count_coll_pen: float = -1.0
    max_count_colls: int = -1
    count_coll_end_pen: float = 1.0
    max_target_distance: float = -1.0
    max_target_distance_pen: float = 1.0
    non_desire_ee_local_pos_dis: float = -1.0
    non_desire_ee_local_pos_pen: float = 1.0
    non_desire_ee_local_pos: Optional[List[float]] = None
    camera_looking_down_angle: float = -1.0
    camera_looking_down_pen: float = 1.0


@dataclass
class RearrangePickSuccessMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Only. Requires the end_effector_sensor lab sensor. 1.0 if the robot picked the target object.

    :property ee_resting_success_threshold: -1 if we do not consider resetting the arm to the resting location
    """
    type: str = "RearrangePickSuccess"
    ee_resting_success_threshold: float = 0.15


@dataclass
class ObjAtGoalMeasurementConfig(MeasurementConfig):
    r"""
    The measure is a dictionary of target indexes to float. The values are 1 if the object is within succ_thresh of the goal position for that object.

    :property succ_thresh: The threshold distance below which an object is considered at the goal location.
    """
    type: str = "ObjAtGoal"
    succ_thresh: float = 0.15


@dataclass
class PlaceRewardMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Only. Requires the end_effector_sensor lab sensor. The reward for the place task.

    :property dist_reward: At each step, the measure adds dist_reward times the distance the end effector moved towards the target.
    :property place_reward: If the robot placed the target object correctly, it receives place_reward reward.
    :property drop_pen: The penalty for dropping the object.
    :property force_pen: At each step, adds a penalty of force_pen times the current force on the robot.
    :property wrong_drop_should_end: If true, the task will end if the robot drops the object.
    """
    type: str = "PlaceReward"
    dist_reward: float = 2.0
    place_reward: float = 5.0
    drop_pen: float = 0.0
    use_diff: bool = True
    use_ee_dist: bool = False
    wrong_drop_should_end: bool = True
    constraint_violate_pen: float = 0.0
    force_pen: float = 0.0001
    max_force_pen: float = 0.0
    force_end_pen: float = 1.0
    min_dist_to_goal: float = 0.15
    count_coll_pen: float = -1.0
    max_count_colls: int = -1
    count_coll_end_pen: float = 1.0


@dataclass
class PlaceSuccessMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Only. Requires the end_effector_sensor lab sensor. 1.0 if the robot placed the target object on the goal position and has its end effector within ee_resting_success_threshold of its resting position.
    """
    type: str = "PlaceSuccess"
    ee_resting_success_threshold: float = 0.15


@dataclass
class PddlStageGoalsMeasurementConfig(MeasurementConfig):
    r"""
    PDDL Rearrangement only. 1.0 if the agent complete a particular stage defined in `stage_goals` and 0.0 otherwise. Stage goals are specified in the `pddl` task description.
    """
    type: str = "PddlStageGoals"


@dataclass
class PddlSuccessMeasurementConfig(MeasurementConfig):
    r"""
    PDDL rearrangement tasks only (rearrange, set_table, tidy_house). It uses a goal pddl expression to validate the success.

    :property must_call_stop: If true, the robot must in addition, call the rearrange_stop action for this measure to be a success.
    """
    type: str = "PddlSuccess"
    must_call_stop: bool = True


@dataclass
class PddlSubgoalReward(MeasurementConfig):
    type: str = "PddlSubgoalReward"
    stage_sparse_reward: float = 1.0


@dataclass
class DidAgentsCollideConfig(MeasurementConfig):
    type: str = "DidAgentsCollide"


@dataclass
class NumAgentsCollideConfig(MeasurementConfig):
    type: str = "NumAgentsCollide"


@dataclass
class RearrangeCooperateRewardConfig(PddlSubgoalReward):
    type: str = "RearrangeCooperateReward"
    stage_sparse_reward: float = 1.0
    end_on_collide: bool = True
    # Positive penalty means give negative reward.
    collide_penalty: float = 1.0


@dataclass
class CooperateSubgoalRewardConfig(PddlSubgoalReward):
    type: str = "CooperateSubgoalReward"
    stage_sparse_reward: float = 1.0
    end_on_collide: bool = True
    # Positive penalty means give negative reward.
    collide_penalty: float = 1.0


@dataclass
class SocialNavReward(MeasurementConfig):
    r"""
    The reward for the social navigation tasks.
    """
    type: str = "SocialNavReward"
    # The safety distance between the robot and the human
    safe_dis_min: float = 1.0
    safe_dis_max: float = 2.0
    # If the safety distance is within the threshold, then
    # the agent receives this amount of reward
    safe_dis_reward: float = 2.0
    # If the distance is below this threshold, the robot
    # starts receiving an additional orientation reward
    facing_human_dis: float = 3.0
    # -1 means that there is no facing_human_reward
    facing_human_reward: float = -1.0
    # toward_human_reward default is 1.0
    toward_human_reward: float = 1.0
    # -1 means that there is no near_human_bonus
    near_human_bonus: float = -1.0
    # -1 means that there is no exploration reward
    explore_reward: float = -1.0
    # If we want to use geo distance to measure the distance
    # between the robot and the human
    use_geo_distance: bool = False
    # Set the id of the agent
    robot_idx: int = 0
    human_idx: int = 1
    constraint_violate_pen: float = 10.0
    force_pen: float = 0.0
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0
    # Collision based penality for kinematic simulation
    count_coll_pen: float = -1.0
    max_count_colls: int = -1
    count_coll_end_pen: float = 1.0
    collide_penalty: float = 1.0


@dataclass
class DoesWantTerminateMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Only. Measures 1 if the agent has called the stop action and 0 otherwise.
    """
    type: str = "DoesWantTerminate"


@dataclass
class CorrectAnswerMeasurementConfig(MeasurementConfig):
    type: str = "CorrectAnswer"


@dataclass
class EpisodeInfoMeasurementConfig(MeasurementConfig):
    type: str = "EpisodeInfo"


@dataclass
class DistanceToGoalMeasurementConfig(MeasurementConfig):
    r"""
    In Navigation tasks only, measures the geodesic distance to the goal.

    :property distance_to: If 'POINT' measures the distance to the closest episode goal. If 'VIEW_POINTS' measures the distance to the episode's goal's viewpoint.
    """
    type: str = "DistanceToGoal"
    distance_to: str = "POINT"


@dataclass
class DistanceToGoalRewardMeasurementConfig(MeasurementConfig):
    r"""
    In Navigation tasks only, measures a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """
    type: str = "DistanceToGoalReward"


@dataclass
class AnswerAccuracyMeasurementConfig(MeasurementConfig):
    type: str = "AnswerAccuracy"


@dataclass
class TaskConfig(HabitatBaseConfig):
    r"""
    The definition of the task in Habitat.

    :property type: The registered task that will be used. For example : `InstanceImageNav-v1` or `ObjectNav-v1`
    :property physics_target_sps: The size of each simulator physics update will be 1 / physics_target_sps.
    :property reward_measure: The name of the Measurement that will correspond to the reward of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration. For example, `distance_to_goal_reward` for navigation or `place_reward` for the rearrangement place task.
    :property success_measure: The name of the Measurement that will correspond to the success criteria of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration. If the measurement has a non-zero value, the episode is considered a success.
    :property end_on_success: If True, the episode will end when the success measure indicates success. Otherwise the episode will go on (this is useful when doing hierarchical learning and the robot has to explicitly decide when to change policies)
    :property task_spec: When doing the `RearrangePddlTask-v0` only, will look for a pddl plan of that name to determine the sequence of tasks that need to be completed. The format of the pddl plans files is undocumented.
    :property task_spec_base_path:  When doing the `RearrangePddlTask-v0` only, the relative path where the task_spec file will be searched.
    :property spawn_max_dists_to_obj: For `RearrangePickTask-v0` task only. Controls the maximum distance the robot can be spawned from the target object.
    :property base_angle_noise: For Rearrangement tasks only. Controls the standard deviation of the random normal noise applied to the base's rotation angle at the start of an episode.
    :property base_noise: For Rearrangement tasks only. Controls the standard deviation of the random normal noise applied to the base's position at the start of an episode.

    There are many different Tasks determined by the `habitat.task.type` config:

    -   Point navigation : `Nav-0`
    -   Image navigation : `Nav-0`
    -   Instance image navigation:`InstanceImageNav-v1`
    -   Object navigation : `ObjectNav-v1`
    -   Rearrangement close drawer: `RearrangeCloseDrawerTask-v0`
    -   Rearrangement open drawer: `RearrangeOpenDrawerTask-v0`
    -   Rearrangement close fridge : `RearrangeCloseFridgeTask-v0`
    -   Rearrangement open fridge : `RearrangeOpenFridgeTask-v0`
    -   Rearrangement navigate to object : `NavToObjTask-v0`
    -   Rearrangement pick : `RearrangePickTask-v0`
    -   Rearrangement place : `RearrangePlaceTask-v0`
    -   Rearrangement do nothing : `RearrangeEmptyTask-v0`
    -   Rearrangement reach : `RearrangeReachTask-v0`
    -   Rearrangement composite tasks : `RearrangePddlTask-v0`
    """
    physics_target_sps: float = 60.0
    reward_measure: Optional[str] = None
    success_measure: Optional[str] = None
    success_reward: float = 2.5
    slack_reward: float = -0.01
    end_on_success: bool = False
    # NAVIGATION task
    type: str = "Nav-v0"
    # Temporary structure for sensors
    lab_sensors: Dict[str, LabSensorConfig] = field(default_factory=dict)
    measurements: Dict[str, MeasurementConfig] = field(default_factory=dict)
    # Measures to only construct in the first environment of the first rank for
    # vectorized environments.
    rank0_env0_measure_names: List[str] = field(
        default_factory=lambda: ["habitat_perf"]
    )
    # Measures to only record in the first rank for vectorized environments.
    rank0_measure_names: List[str] = field(default_factory=list)
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
    object_in_hand_sample_prob: float = 0.167
    min_start_distance: float = 3.0
    gfx_replay_dir = "data/replays"
    render_target: bool = True
    # Spawn parameters
    filter_colliding_states: bool = True
    num_spawn_attempts: int = 200
    spawn_max_dist_to_obj: float = 2.0
    base_angle_noise: float = 0.523599
    spawn_max_dist_to_obj_delta: float = 0.02
    # Factor to shrink the receptacle sampling volume when predicates place
    # objects on top of receptacles.
    recep_place_shrink_factor: float = 0.8
    # EE sample parameters
    ee_sample_factor: float = 0.2
    ee_exclude_region: float = 0.0
    base_noise: float = 0.05
    spawn_region_scale: float = 0.2
    joint_max_impulse: float = -1.0
    desired_resting_position: List[float] = field(
        default_factory=lambda: [0.5, 0.0, 1.0]
    )
    use_marker_t: bool = True
    cache_robot_init: bool = False
    success_state: float = 0.0
    # Measurements for composite tasks.
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

    # The minimum distance between the agents at start. If < 0
    # there is no minimal distance
    min_distance_start_agents: float = -1.0
    actions: Dict[str, ActionConfig] = MISSING


@dataclass
class SimulatorSensorConfig(HabitatBaseConfig):
    type: str = MISSING
    height: int = 480
    width: int = 640
    position: List[float] = field(default_factory=lambda: [0.0, 1.25, 0.0])
    # Euler's angles:
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class SimulatorCameraSensorConfig(SimulatorSensorConfig):
    hfov: int = 90  # horizontal field of view in degrees
    sensor_subtype: str = "PINHOLE"
    noise_model: str = "None"
    noise_model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorDepthSensorConfig(SimulatorSensorConfig):
    min_depth: float = 0.0
    max_depth: float = 10.0
    normalize_depth: bool = True


@dataclass
class HabitatSimRGBSensorConfig(SimulatorCameraSensorConfig):
    type: str = "HabitatSimRGBSensor"


@dataclass
class HabitatSimDepthSensorConfig(SimulatorCameraSensorConfig):
    type: str = "HabitatSimDepthSensor"
    min_depth: float = 0.0
    max_depth: float = 10.0
    normalize_depth: bool = True


@dataclass
class HabitatSimSemanticSensorConfig(SimulatorCameraSensorConfig):
    type: str = "HabitatSimSemanticSensor"


@dataclass
class HabitatSimEquirectangularRGBSensorConfig(SimulatorSensorConfig):
    type: str = "HabitatSimEquirectangularRGBSensor"


@dataclass
class HabitatSimEquirectangularDepthSensorConfig(SimulatorDepthSensorConfig):
    type: str = "HabitatSimEquirectangularDepthSensor"


@dataclass
class HabitatSimEquirectangularSemanticSensorConfig(SimulatorSensorConfig):
    type: str = "HabitatSimEquirectangularSemanticSensor"


@dataclass
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
    focal_length: List[float] = field(default_factory=lambda: [364.84, 364.86])
    # Place camera at center of screen
    # Can be specified, otherwise is calculated automatically.
    # principal_point_offset defaults to (h/2,w/2)
    principal_point_offset: Optional[List[float]] = None
    sensor_model_type: str = "DOUBLE_SPHERE"


@dataclass
class HabitatSimFisheyeRGBSensorConfig(SimulatorFisheyeSensorConfig):
    type: str = "HabitatSimFisheyeRGBSensor"


@dataclass
class SimulatorFisheyeDepthSensorConfig(SimulatorFisheyeSensorConfig):
    type: str = "HabitatSimFisheyeDepthSensor"
    min_depth: float = SimulatorDepthSensorConfig().min_depth
    max_depth: float = SimulatorDepthSensorConfig().max_depth
    normalize_depth: bool = SimulatorDepthSensorConfig().normalize_depth


@dataclass
class HabitatSimFisheyeSemanticSensorConfig(SimulatorFisheyeSensorConfig):
    type: str = "HabitatSimFisheyeSemanticSensor"


@dataclass
class HeadRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "head_rgb"
    width: int = 256
    height: int = 256


@dataclass
class HeadDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "head_depth"
    width: int = 256
    height: int = 256


@dataclass
class HeadStereoLeftDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "head_stereo_left_depth"
    width: int = 256
    height: int = 256


@dataclass
class HeadStereoRightDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "head_stereo_right_depth"
    width: int = 256
    height: int = 256


@dataclass
class HeadPanopticSensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "head_panoptic"
    width: int = 256
    height: int = 256


@dataclass
class ArmPanopticSensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "articulated_agent_arm_panoptic"
    width: int = 256
    height: int = 256


@dataclass
class JawPanopticSensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "articulated_agent_jaw_panoptic"
    width: int = 256
    height: int = 256


@dataclass
class ArmRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "articulated_agent_arm_rgb"
    width: int = 256
    height: int = 256


@dataclass
class ArmDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "articulated_agent_arm_depth"
    width: int = 256
    height: int = 256


@dataclass
class JawRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "articulated_agent_jaw_rgb"
    width: int = 256
    height: int = 256


@dataclass
class JawDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "articulated_agent_jaw_depth"
    width: int = 256
    height: int = 256


@dataclass
class ThirdRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "third_rgb"
    width: int = 512
    height: int = 512


@dataclass
class ThirdDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "third_depth"  # TODO: third_rgb on the main branch
    #  check if it won't cause any errors


@dataclass
class AgentConfig(HabitatBaseConfig):
    height: float = 1.5
    radius: float = 0.1
    max_climb: float = 0.2
    max_slope: float = 45.0
    grasp_managers: int = 1
    sim_sensors: Dict[str, Any] = field(default_factory=dict)
    is_set_start_state: bool = False
    start_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    start_rotation: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    joint_start_noise: float = 0.1
    joint_that_can_control: Optional[List[int]] = None
    # Hard-code the robot joint start. `joint_start_noise` still applies.
    joint_start_override: Optional[List[float]] = None
    articulated_agent_urdf: Optional[str] = None
    articulated_agent_type: Optional[str] = None
    ik_arm_urdf: Optional[str] = None
    # File to motion data, used to play pre-recorded motions
    motion_data_path: str = ""
    auto_update_sensor_transform: bool = True
    """
    If `True`, the agent's sensor transforms are automatically updated every frame.
    """
    # TODO: Remove this flag once sensors are decoupled from agents.


@dataclass
class RendererConfig(HabitatBaseConfig):
    r"""Configuration for the renderer.

    :property enable_batch_renderer: [Experimental] Enables batch rendering, which accelerates rendering for concurrent environments. See env_batch_renderer.py for details.
    :property composite_files: List of composite GLTF files to be pre-loaded by the batch renderer.
    :property classic_replay_renderer: For debugging. Create a ClassicReplayRenderer instead of BatchReplayRenderer when enable_batch_renderer is active.
    """

    enable_batch_renderer: bool = False
    composite_files: Optional[List[str]] = None
    classic_replay_renderer: bool = False


@dataclass
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
    enable_hbao: bool = False
    physics_config_file: str = "./data/default.physics_config.json"
    # Possibly unstable optimization for extra performance
    # with concurrent rendering
    leave_context_with_background_renderer: bool = False
    enable_gfx_replay_save: bool = False


@dataclass
class SimulatorConfig(HabitatBaseConfig):
    type: str = "Sim-v0"
    forward_step_size: float = 0.25  # in metres
    turn_angle: int = 10  # angle to rotate left or right in degrees
    create_renderer: bool = False
    requires_textures: bool = True
    # Sleep options
    auto_sleep: bool = False
    step_physics: bool = True
    concur_render: bool = False
    # If markers should be updated at every step:
    needs_markers: bool = True
    # If the articulated_agent camera positions should be updated at every step:
    update_articulated_agent: bool = True
    scene: str = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    # The scene dataset to load in the metadatamediator,
    # should contain simulator.scene:
    scene_dataset: str = "default"
    # A list of directory or config paths to search in addition to the dataset
    # for object configs. should match the generated episodes for the task:
    additional_object_paths: List[str] = field(default_factory=list)
    # Use config.seed (can't reference Config.seed) or define via code
    # otherwise it leads to circular references:
    #
    seed: int = II("habitat.seed")
    default_agent_id: int = 0
    debug_render: bool = False
    debug_render_articulated_agent: bool = False
    kinematic_mode: bool = False
    # If False, will skip setting the semantic IDs of objects in
    # `rearrange_sim.py` (there is overhead to this operation so skip if not
    # using semantic information).
    should_setup_semantic_ids: bool = True
    # If in render mode a visualization of the rearrangement goal position
    # should also be displayed
    debug_render_goal: bool = True
    robot_joint_start_noise: float = 0.0
    # Rearrange agent setup
    ctrl_freq: float = 120.0
    ac_freq_ratio: int = 4
    load_objs: bool = True
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

    # Simulator should use default navmesh settings from agent config
    default_agent_navmesh: bool = True
    # if default navmesh is used, should it include static objects
    navmesh_include_static_objects: bool = False

    habitat_sim_v0: HabitatSimV0Config = HabitatSimV0Config()
    # ep_info is added to the config in some rearrange tasks inside
    # merge_sim_episode_with_object_config
    ep_info: Optional[Any] = None
    # The offset id values for the object
    object_ids_start: int = 100
    # Configuration for rendering
    renderer: RendererConfig = RendererConfig()


@dataclass
class LocobotConfig(HabitatBaseConfig):
    actions: List[str] = field(
        default_factory=lambda: ["base_actions", "camera_actions"]
    )
    base_actions: List[str] = field(
        default_factory=lambda: ["go_to_relative", "go_to_absolute"]
    )
    camera_actions: List[str] = field(
        default_factory=lambda: ["set_pan", "set_tilt", "set_pan_tilt"]
    )


@dataclass
class DatasetConfig(HabitatBaseConfig):
    r"""Configuration for the dataset of the task.

    :property type: The key for the dataset class that will be used. Examples of such keys are `PointNav-v1`, `ObjectNav-v1`, `InstanceImageNav-v1` or `RearrangeDataset-v0`. Different datasets have different properties so you should use the dataset that fits your task.
    :property scenes_dir: The path to the directory containing the scenes that will be used. You should put all your scenes in the same folder (example `data/scene_datasets`) to avoid having to change it.
    :property data_path: The path to the episode dataset. Episodes need to be compatible with the `type` argument (so they will load properly) and only use scenes that are present in the `scenes_dir`.
    :property split: `data_path` can have a `split` in the path. For example: "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz" the value in "{split}" will be replaced by the value of the `split` argument. This allows to easily swap between training, validation and test episodes by only changing the split argument.
    :property metadata: Optional. Additional information for interpreting the dataset.

    A dataset consists of episodes
    (a start configuration for a task within a scene) and a scene dataset
    (with all the assets needed to instantiate the task)
    """
    type: str = "PointNav-v1"
    split: str = "train"
    scenes_dir: str = "data/scene_datasets"
    content_scenes: List[str] = field(default_factory=lambda: ["*"])
    data_path: str = (
        "data/datasets/pointnav/"
        "habitat-test-scenes/v1/{split}/{split}.json.gz"
    )
    # TODO: Make this field a structured dataclass.
    metadata: Optional[Any] = None


@dataclass
class GymConfig(HabitatBaseConfig):
    obs_keys: Optional[List[str]] = None
    action_keys: Optional[List[str]] = None
    achieved_goal_keys: List = field(default_factory=list)
    desired_goal_keys: List[str] = field(default_factory=list)


@dataclass
class HabitatConfig(HabitatBaseConfig):
    r"""
    The entry point for the configuration of Habitat. It holds the environment, simulator, task and dataset configurations.

    :property seed: The seed the environment will be initialized with.
    :property env_task: Indicates wether the environment is a Habitat gym environment (`GymHabitatEnv`) or a generic gym environment (`GymRegistryEnv`).
    :property env_task_gym_id: if `env_task` is `GymRegistryEnv`, env_task_gym_id is the identifier of the generic gym environment
    """
    seed: int = 100
    # GymHabitatEnv works for all Habitat tasks, including Navigation and
    # Rearrange. To use a gym environment from the registry, use the
    # GymRegistryEnv. Any other environment needs to be created and registered.
    env_task: str = "GymHabitatEnv"
    # The dependencies for launching the GymRegistryEnv environments.
    # Modules listed here will be imported prior to making the environment with
    # gym.make()
    env_task_gym_dependencies: List = field(default_factory=list)
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
    package="habitat.task.actions.base_velocity_non_cylinder",
    group="habitat/task/actions",
    name="base_velocity_non_cylinder",
    node=BaseVelocityNonCylinderActionConfig,
)
cs.store(
    package="habitat.task.actions.humanoidjoint_action",
    group="habitat/task/actions",
    name="humanoidjoint_action",
    node=HumanoidJointActionConfig,
)
cs.store(
    package="habitat.task.actions.humanoid_pick_action",
    group="habitat/task/actions",
    name="humanoid_pick_action",
    node=HumanoidPickActionConfig,
)
cs.store(
    package="habitat.task.actions.velocity_control",
    group="habitat/task/actions",
    name="velocity_control",
    node=VelocityControlActionConfig,
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
    package="habitat.task.actions.a_selection_of_base_or_arm",
    group="habitat/task/actions",
    name="a_selection_of_base_or_arm",
    node=SelectBaseOrArmActionConfig,
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
cs.store(
    package="habitat.task.actions.pddl_apply_action",
    group="habitat/task/actions",
    name="pddl_apply_action",
    node=PddlApplyActionConfig,
)

# Dataset Config Schema
cs.store(
    package="habitat.dataset",
    group="habitat/dataset",
    name="dataset_config_schema",
    node=DatasetConfig,
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
    name="jaw_depth_sensor",
    node=JawDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="jaw_rgb_sensor",
    node=JawRGBSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="head_depth_sensor",
    node=HeadDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="head_stereo_right_depth_sensor",
    node=HeadStereoRightDepthSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="head_stereo_left_depth_sensor",
    node=HeadStereoLeftDepthSensorConfig,
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
    name="arm_panoptic_sensor",
    node=ArmPanopticSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="jaw_panoptic_sensor",
    node=JawPanopticSensorConfig,
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
    package="habitat.task.lab_sensors.pointgoal_with_gps_compass_sensor",
    group="habitat/task/lab_sensors",
    name="pointgoal_with_gps_compass_sensor",
    node=PointGoalWithGPSCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.humanoid_detector_sensor",
    group="habitat/task/lab_sensors",
    name="humanoid_detector_sensor",
    node=HumanoidDetectorSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.arm_depth_bbox_sensor",
    group="habitat/task/lab_sensors",
    name="arm_depth_bbox_sensor",
    node=ArmDepthBBoxSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.spot_head_stereo_depth_sensor",
    group="habitat/task/lab_sensors",
    name="spot_head_stereo_depth_sensor",
    node=SpotHeadStereoDepthSensorConfig,
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
    package="habitat.task.lab_sensors.localization_sensor",
    group="habitat/task/lab_sensors",
    name="localization_sensor",
    node=LocalizationSensorConfig,
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
    package="habitat.task.lab_sensors.humanoid_joint_sensor",
    group="habitat/task/lab_sensors",
    name="humanoid_joint_sensor",
    node=HumanoidJointSensorConfig,
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
    package="habitat.task.lab_sensors.initial_gps_compass_sensor",
    group="habitat/task/lab_sensors",
    name="initial_gps_compass_sensor",
    node=InitialGpsCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.multi_agent_all_predicates",
    group="habitat/task/lab_sensors",
    name="multi_agent_all_predicates",
    node=MultiAgentGlobalPredicatesSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.agents_within_threshold",
    group="habitat/task/lab_sensors",
    name="agents_within_threshold",
    node=AreAgentsWithinThresholdConfig,
)
cs.store(
    package="habitat.task.lab_sensors.has_finished_oracle_nav",
    group="habitat/task/lab_sensors",
    name="has_finished_oracle_nav",
    node=HasFinishedOracleNavSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.has_finished_humanoid_pick",
    group="habitat/task/lab_sensors",
    name="has_finished_humanoid_pick",
    node=HasFinishedHumanoidPickSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.other_agent_gps",
    group="habitat/task/lab_sensors",
    name="other_agent_gps",
    node=OtherAgentGpsConfig,
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
    package="habitat.task.lab_sensors.all_predicates",
    group="habitat/task/lab_sensors",
    name="all_predicates",
    node=GlobalPredicatesSensorConfig,
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
    package="habitat.task.measurements.zero",
    group="habitat/task/measurements",
    name="zero",
    node=ZeroMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.articulated_agent_force",
    group="habitat/task/measurements",
    name="articulated_agent_force",
    node=RobotForceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.force_terminate",
    group="habitat/task/measurements",
    name="force_terminate",
    node=ForceTerminateMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.end_effector_to_object_distance",
    group="habitat/task/measurements",
    name="end_effector_to_object_distance",
    node=EndEffectorToObjectDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.base_to_object_distance",
    group="habitat/task/measurements",
    name="base_to_object_distance",
    node=BaseToObjectDistanceMeasurementConfig,
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
    package="habitat.task.measurements.episode_info",
    group="habitat/task/measurements",
    name="episode_info",
    node=EpisodeInfoMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.collisions",
    group="habitat/task/measurements",
    name="collisions",
    node=CollisionsMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.articulated_agent_colls",
    group="habitat/task/measurements",
    name="articulated_agent_colls",
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
    package="habitat.task.measurements.place_success",
    group="habitat/task/measurements",
    name="place_success",
    node=PlaceSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.place_reward",
    group="habitat/task/measurements",
    name="place_reward",
    node=PlaceRewardMeasurementConfig,
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
    package="habitat.task.measurements.pddl_subgoal_reward",
    group="habitat/task/measurements",
    name="pddl_subgoal_reward",
    node=PddlSubgoalReward,
)
cs.store(
    package="habitat.task.measurements.rearrange_cooperate_reward",
    group="habitat/task/measurements",
    name="rearrange_cooperate_reward",
    node=RearrangeCooperateRewardConfig,
)
cs.store(
    package="habitat.task.measurements.social_nav_reward",
    group="habitat/task/measurements",
    name="social_nav_reward",
    node=SocialNavReward,
)
cs.store(
    package="habitat.task.measurements.did_agents_collide",
    group="habitat/task/measurements",
    name="did_agents_collide",
    node=DidAgentsCollideConfig,
)
cs.store(
    package="habitat.task.measurements.num_agents_collide",
    group="habitat/task/measurements",
    name="num_agents_collide",
    node=NumAgentsCollideConfig,
)
cs.store(
    package="habitat.task.measurements.pddl_success",
    group="habitat/task/measurements",
    name="pddl_success",
    node=PddlSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.gfx_replay_measure",
    group="habitat/task/measurements",
    name="gfx_replay_measure",
    node=GfxReplayMeasureMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pddl_stage_goals",
    group="habitat/task/measurements",
    name="composite_stage_goals",
    node=PddlStageGoalsMeasurementConfig,
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
    package="habitat.task.measurements.social_nav_stats",
    group="habitat/task/measurements",
    name="social_nav_stats",
    node=SocialNavStatsMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.social_nav_seek_success",
    group="habitat/task/measurements",
    name="social_nav_seek_success",
    node=NavSeekSuccessMeasurementConfig,
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
cs.store(
    package="habitat.task.measurements.habitat_perf",
    group="habitat/task/measurements",
    name="habitat_perf",
    node=RuntimePerfStatsMeasurementConfig,
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
