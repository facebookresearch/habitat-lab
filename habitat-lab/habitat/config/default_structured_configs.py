from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


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
    type: str = "StopAction"


@dataclass
class EmptyActionConfig(ActionConfig):
    type: str = "EmptyAction"


# -----------------------------------------------------------------------------
# # NAVIGATION actions
# -----------------------------------------------------------------------------
@dataclass
class MoveForwardActionConfig(ActionConfig):
    type: str = "MoveForwardAction"


@dataclass
class TurnLeftActionConfig(ActionConfig):
    type: str = "TurnLeftAction"


@dataclass
class TurnRightActionConfig(ActionConfig):
    type: str = "TurnRightAction"


@dataclass
class LookUpActionConfig(ActionConfig):
    type: str = "LookUpAction"


@dataclass
class LookDownActionConfig(ActionConfig):
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
    type: str = "ArmAction"
    arm_controller: str = "ArmRelPosAction"
    grip_controller: Optional[str] = None
    arm_joint_dimensionality: int = 7
    grasp_thresh_dist: float = 0.15
    disable_grip: bool = False
    delta_pos_limit: float = 0.0125
    ee_ctrl_lim: float = 0.015
    should_clip: bool = False
    render_ee_target: bool = False


@dataclass
class BaseVelocityActionConfig(ActionConfig):
    type: str = "BaseVelAction"
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    end_on_stop: bool = False
    allow_back: bool = True
    min_abs_lin_speed: float = 1.0
    min_abs_ang_speed: float = 1.0


@dataclass
class RearrangeStopActionConfig(ActionConfig):
    type: str = "RearrangeStopAction"


@dataclass
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
    min_abs_lin_speed: float = 1.0
    min_abs_ang_speed: float = 1.0
    allow_dyn_slide: bool = True
    end_on_stop: bool = False
    allow_back: bool = True


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
class SensorConfig(HabitatBaseConfig):
    type: str = MISSING


@dataclass
class PointGoalSensorConfig(SensorConfig):
    type: str = "PointGoalSensor"
    goal_format: str = "POLAR"
    dimensionality: int = 2


@dataclass
class PointGoalWithGPSCompassSensorConfig(PointGoalSensorConfig):
    type: str = "PointGoalWithGPSCompassSensor"


class ObjectGoalSensorConfig(SensorConfig):
    type: str = "ObjectGoalSensor"
    goal_spec: str = "TASK_CATEGORY_ID"
    goal_spec_max_val: int = 50


@dataclass
class ImageGoalSensorConfig(SensorConfig):
    type: str = "ImageGoalSensor"


@dataclass
class InstanceImageGoalSensorConfig(SensorConfig):
    type: str = "InstanceImageGoalSensor"


@dataclass
class InstanceImageGoalHFOVSensorConfig(SensorConfig):
    type: str = "InstanceImageGoalHFOVSensor"


@dataclass
class HeadingSensorConfig(SensorConfig):
    type: str = "HeadingSensor"


@dataclass
class CompassSensorConfig(SensorConfig):
    type: str = "CompassSensor"


@dataclass
class GPSSensorSensorConfig(SensorConfig):
    type: str = "GPSSensor"
    dimensionality: int = 2


@dataclass
class ProximitySensorConfig(SensorConfig):
    type: str = "ProximitySensor"
    max_detection_radius: float = 2.0


@dataclass
class JointSensorConfig(SensorConfig):
    type: str = "JointSensor"
    dimensionality: int = 7


@dataclass
class EEPositionSensorConfig(SensorConfig):
    type: str = "EEPositionSensor"


@dataclass
class IsHoldingSensorConfig(SensorConfig):
    type: str = "IsHoldingSensor"


@dataclass
class RelativeRestingPositionSensorConfig(SensorConfig):
    type: str = "RelativeRestingPositionSensor"


@dataclass
class JointVelocitySensorConfig(SensorConfig):
    type: str = "JointVelocitySensor"
    dimensionality: int = 7


@dataclass
class OracleNavigationActionSensorConfig(SensorConfig):
    type: str = "OracleNavigationActionSensor"


@dataclass
class RestingPositionSensorConfig(SensorConfig):
    type: str = "RestingPositionSensor"


@dataclass
class ArtJointSensorConfig(SensorConfig):
    type: str = "ArtJointSensor"


@dataclass
class NavGoalSensorConfig(SensorConfig):
    type: str = "NavGoalSensor"


@dataclass
class ArtJointSensorNoVelSensorConfig(SensorConfig):
    type: str = "ArtJointSensorNoVel"  # TODO: add "Sensor" suffix


@dataclass
class MarkerRelPosSensorConfig(SensorConfig):
    type: str = "MarkerRelPosSensor"


@dataclass
class TargetStartSensorConfig(SensorConfig):
    type: str = "TargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class TargetCurrentSensorConfig(SensorConfig):
    type: str = "TargetCurrentSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class GoalSensorConfig(SensorConfig):
    type: str = "GoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class TargetOrGoalStartPointGoalSensorConfig(SensorConfig):
    type: str = "TargetOrGoalStartPointGoalSensor"


@dataclass
class GlobalPredicatesSensorConfig(SensorConfig):
    type: str = "GlobalPredicatesSensor"


@dataclass
class TargetStartGpsCompassSensorConfig(SensorConfig):
    type: str = "TargetStartGpsCompassSensor"


@dataclass
class TargetGoalGpsCompassSensorConfig(SensorConfig):
    type: str = "TargetGoalGpsCompassSensor"


@dataclass
class NavToSkillSensorConfig(SensorConfig):
    type: str = "NavToSkillSensor"
    num_skills: int = 8


@dataclass
class AbsTargetStartSensorConfig(SensorConfig):
    type: str = "AbsTargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class AbsGoalSensorConfig(SensorConfig):
    type: str = "AbsGoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class DistToNavGoalSensorConfig(SensorConfig):
    type: str = "DistToNavGoalSensor"


@dataclass
class LocalizationSensorConfig(SensorConfig):
    type: str = "LocalizationSensor"


@dataclass
class QuestionSensorConfig(SensorConfig):
    type: str = "QuestionSensor"


@dataclass
class InstructionSensorConfig(SensorConfig):
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
    type: str = "Success"
    success_distance: float = 0.2


@dataclass
class SPLMeasurementConfig(MeasurementConfig):
    type: str = "SPL"


@dataclass
class SoftSPLMeasurementConfig(MeasurementConfig):
    type: str = "SoftSPL"


@dataclass
class FogOfWarConfig:
    draw: bool = True
    visibility_dist: float = 5.0
    fov: int = 90


@dataclass
class TopDownMapMeasurementConfig(MeasurementConfig):
    type: str = "TopDownMap"
    max_episode_steps: int = EnvironmentConfig.max_episode_steps
    map_padding: int = 3
    map_resolution: int = 1024
    draw_source: bool = True
    draw_border: bool = True
    draw_shortest_path: bool = True
    draw_view_points: bool = True
    draw_goal_positions: bool = True
    # axes aligned bounding boxes
    draw_goal_aabbs: bool = True
    fog_of_war = FogOfWarConfig()


@dataclass
class CollisionsMeasurementConfig(MeasurementConfig):
    type: str = "Collisions"


@dataclass
class RobotForceMeasurementConfig(MeasurementConfig):
    type: str = "RobotForce"
    min_force: float = 20.0


@dataclass
class ForceTerminateMeasurementConfig(MeasurementConfig):
    type: str = "ForceTerminate"
    max_accum_force: float = -1.0
    max_instant_force: float = -1.0


@dataclass
class RobotCollisionsMeasurementConfig(MeasurementConfig):
    type: str = "RobotCollisions"


@dataclass
class ObjectToGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "ObjectToGoalDistance"


@dataclass
class EndEffectorToObjectDistanceMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorToObjectDistance"


@dataclass
class EndEffectorToRestDistanceMeasurementConfig(MeasurementConfig):
    type: str = "EndEffectorToRestDistance"


@dataclass
class ArtObjAtDesiredStateMeasurementConfig(MeasurementConfig):
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
    type: str = "ArtObjSuccess"
    rest_dist_threshold: float = 0.15


@dataclass
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
    force_pen: float = 0.0
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0


@dataclass
class RotDistToGoalMeasurementConfig(MeasurementConfig):
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
    type: str = "NavToPosSucc"
    success_distance: float = 0.2


@dataclass
class NavToObjRewardMeasurementConfig(MeasurementConfig):
    type: str = "NavToObjReward"
    # reward the agent for facing the object?
    should_reward_turn: bool = True
    # what distance do we start giving the reward for facing the object?
    turn_reward_dist: float = 0.1
    # multiplier on the angle distance to the goal.
    angle_dist_reward: float = 1.0
    dist_reward: float = 10.0
    constraint_violate_pen: float = 10.0
    force_pen: float = 0.0
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0


@dataclass
class NavToObjSuccessMeasurementConfig(MeasurementConfig):
    type: str = "NavToObjSuccess"
    must_look_at_targ: bool = True
    must_call_stop: bool = True
    # distance in radians.
    success_angle_dist: float = 0.15
    heuristic_stop: bool = False


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
    type: str = "NumStepsMeasure"


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


@dataclass
class RearrangePickRewardMeasurementConfig(MeasurementConfig):
    type: str = "RearrangePickReward"
    dist_reward: float = 20.0
    succ_reward: float = 10.0
    pick_reward: float = 20.0
    constraint_violate_pen: float = 10.0
    drop_pen: float = 5.0
    wrong_pick_pen: float = 5.0
    max_accum_force: float = 5000.0
    force_pen: float = 0.001
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0
    use_diff: bool = True
    drop_obj_should_end: bool = False
    wrong_pick_should_end: bool = False


@dataclass
class RearrangePickSuccessMeasurementConfig(MeasurementConfig):
    type: str = "RearrangePickSuccess"
    ee_resting_success_threshold: float = 0.15


@dataclass
class ObjAtGoalMeasurementConfig(MeasurementConfig):
    type: str = "ObjAtGoal"
    succ_thresh: float = 0.15


@dataclass
class PlaceRewardMeasurementConfig(MeasurementConfig):
    type: str = "PlaceReward"
    dist_reward: float = 20.0
    succ_reward: float = 10.0
    place_reward: float = 20.0
    drop_pen: float = 5.0
    use_diff: bool = True
    wrong_drop_should_end: bool = False
    constraint_violate_pen: float = 10.0
    force_pen: float = 0.001
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0


@dataclass
class PlaceSuccessMeasurementConfig(MeasurementConfig):
    type: str = "PlaceSuccess"
    ee_resting_success_threshold: float = 0.15


@dataclass
class CompositeNodeIdxMeasurementConfig(MeasurementConfig):
    type: str = "CompositeNodeIdx"


@dataclass
class CompositeStageGoalsMeasurementConfig(MeasurementConfig):
    type: str = "CompositeStageGoals"


@dataclass
class CompositeSuccessMeasurementConfig(MeasurementConfig):
    type: str = "CompositeSuccess"
    must_call_stop: bool = True


@dataclass
class CompositeRewardMeasurementConfig(MeasurementConfig):
    type: str = "CompositeReward"
    must_call_stop: bool = True
    success_reward: float = 10.0


@dataclass
class DoesWantTerminateMeasurementConfig(MeasurementConfig):
    type: str = "DoesWantTerminate"


@dataclass
class CompositeBadCalledTerminateMeasurementConfig(MeasurementConfig):
    type: str = "CompositeBadCalledTerminate"


@dataclass
class CorrectAnswerMeasurementConfig(MeasurementConfig):
    type: str = "CorrectAnswer"


@dataclass
class EpisodeInfoMeasurementConfig(MeasurementConfig):
    type: str = "EpisodeInfo"


@dataclass
class DistanceToGoalMeasurementConfig(MeasurementConfig):
    type: str = "DistanceToGoal"
    distance_to: str = "POINT"


@dataclass
class DistanceToGoalRewardMeasurementConfig(MeasurementConfig):
    type: str = "DistanceToGoalReward"


@dataclass
class AnswerAccuracyMeasurementConfig(MeasurementConfig):
    type: str = "AnswerAccuracy"


@dataclass
class TaskConfig(HabitatBaseConfig):
    reward_measure: Optional[str] = None
    success_measure: Optional[str] = None
    success_reward: float = 2.5
    slack_reward: float = -0.01
    end_on_success: bool = False
    # NAVIGATION task
    type: str = "Nav-v0"
    sensors: List[str] = field(default_factory=list)
    measurements: List[str] = field(default_factory=list)
    goal_sensor_uuid: str = "pointgoal"
    possible_actions: List[str] = field(
        default_factory=lambda: [
            "stop",
            "move_forward",
            "turn_left",
            "turn_right",
        ]
    )
    # REARRANGE task
    count_obj_collisions: bool = True
    settle_steps = 5
    constraint_violation_ends_episode = True
    constraint_violation_drops_object = False
    # Forced to regenerate the starts even if they are already cached
    force_regenerate = False
    # Saves the generated starts to a cache if they are not already generated
    should_save_to_cache = True
    must_look_at_targ = True
    object_in_hand_sample_prob = 0.167
    gfx_replay_dir = "data/replays"
    render_target: bool = True
    ee_sample_factor: float = 0.2
    ee_exclude_region: float = 0.0
    # In radians
    base_angle_noise: float = 0.15
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
    # If true, does not care about navigability or collisions
    # with objects when spawning robot
    easy_init: bool = False
    should_enforce_target_within_reach: bool = False
    # COMPOSITE task CONFIG
    task_spec_base_path: str = "configs/tasks/rearrange/pddl/"
    task_spec: str = ""
    # PDDL domain params
    pddl_domain_def: str = "replica_cad"
    obj_succ_thresh: float = 0.3
    art_succ_thresh: float = 0.15
    robot_at_thresh: float = 2.0
    filter_nav_to_tasks: List = field(default_factory=list)
    actions: Any = MISSING


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


@dataclass
class SimulatorDepthSensorConfig(SimulatorSensorConfig):
    min_depth: float = 0.0
    max_depth: float = 10.0
    normalize_depth: bool = True


@dataclass
class HabitatSimRGBSensorConfig(SimulatorCameraSensorConfig):
    type: str = "HabitatSimRGBSensor"


@dataclass
class HabitatSimDepthSensorConfig(SimulatorDepthSensorConfig):
    type: str = "HabitatSimDepthSensor"


@dataclass
class HabitatSimSemanticSensorConfig(SimulatorSensorConfig):
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
    height: int = SimulatorSensorConfig.width
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
    min_depth: float = SimulatorDepthSensorConfig.min_depth
    max_depth: float = SimulatorDepthSensorConfig.max_depth
    normalize_depth: bool = SimulatorDepthSensorConfig.normalize_depth


@dataclass
class HabitatSimFisheyeSemanticSensorConfig(SimulatorFisheyeSensorConfig):
    type: str = "HabitatSimFisheyeSemanticSensor"


@dataclass
class HeadRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "robot_head_rgb"


@dataclass
class HeadDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "robot_head_depth"


@dataclass
class ArmRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "robot_arm_rgb"


@dataclass
class ArmDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "robot_arm_depth"


@dataclass
class ThirdRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "robot_third_rgb"


@dataclass
class ThirdDepthSensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "robot_third_depth"  # TODO: robot_third_rgb on the main branch
    #  check if it won't cause any errors


@dataclass
class AgentConfig(HabitatBaseConfig):
    height: float = 1.5
    radius: float = 0.1
    sensors: List[str] = field(default_factory=lambda: ["rgb_sensor"])
    is_set_start_state: bool = False
    start_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    start_rotation: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    joint_start_noise: float = 0.0
    robot_urdf: str = "data/robots/hab_fetch/robots/hab_fetch.urdf"
    robot_type: str = "FetchRobot"
    ik_arm_urdf: str = "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"


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
    physics_config_file: str = "./data/default.physics_config.json"
    # Possibly unstable optimization for extra performance
    # with concurrent rendering
    leave_context_with_background_renderer: bool = False
    enable_gfx_replay_save: bool = False


@dataclass
class SimulatorConfig(HabitatBaseConfig):
    type: str = "Sim-v0"
    action_space_config: str = "v0"
    forward_step_size: float = 0.25  # in metres
    create_renderer: bool = False
    requires_textures: bool = True
    lag_observations: int = 0
    auto_sleep: bool = False
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
    additional_object_paths: List = field(default_factory=list)
    # Use config.seed (can't reference Config.seed) or define via code
    # otherwise it leads to circular references:
    # seed = Config.seed
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
    hold_thresh: float = 0.09
    grasp_impulse: float = 1000.0
    agents: List[str] = field(default_factory=lambda: ["agent_0"])
    agent_0: AgentConfig = AgentConfig()
    rgb_sensor: HabitatSimRGBSensorConfig = HabitatSimRGBSensorConfig()
    depth_sensor: HabitatSimDepthSensorConfig = HabitatSimDepthSensorConfig()
    habitat_sim_v0: HabitatSimV0Config = HabitatSimV0Config()


@dataclass
class PyrobotConfig(HabitatBaseConfig):
    # types of robots supported:
    robots: List[str] = field(default_factory=lambda: ["locobot"])
    robot: str = "locobot"
    sensors: List[str] = field(
        default_factory=lambda: ["rgb_sensor", "depth_sensor", "bump_sensor"]
    )
    base_controller: str = "proportional"
    base_planner: str = "none"


@dataclass
class PyrobotVisualSensorConfig(HabitatBaseConfig):
    type: str = MISSING
    height: int = 480
    width: int = 640


@dataclass
class PyrobotRGBSensorConfig(PyrobotVisualSensorConfig):
    type: str = "PyRobotRGBSensor"
    center_crop: bool = False


@dataclass
class PyrobotDepthSensorConfig(PyrobotVisualSensorConfig):
    type: str = "PyRobotDepthSensor"
    min_depth: float = 0.0
    max_depth: float = 5.0
    normalize_depth: bool = True
    center_crop: bool = False


@dataclass
class PyrobotBumpSensorConfig(HabitatBaseConfig):
    type: str = "PyRobotBumpSensor"


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
    type: str = "PointNav-v1"
    split: str = "train"
    scenes_dir: str = "data/scene_datasets"
    content_scenes: List[str] = field(default_factory=lambda: ["*"])
    data_path: str = (
        "data/datasets/pointnav/"
        "habitat-test-scenes/v1/{split}/{split}.json.gz"
    )


@dataclass
class GymConfig(HabitatBaseConfig):
    auto_name: str = ""
    obs_keys: Optional[List[str]] = None
    action_keys: Optional[List[str]] = None
    achieved_goal_keys: List = field(default_factory=list)
    desired_goal_keys: List[str] = field(default_factory=list)


@dataclass
class HabitatConfig(HabitatBaseConfig):
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
    task: TaskConfig = TaskConfig()
    dataset: DatasetConfig = DatasetConfig()


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(group="habitat", name="config", node=HabitatConfig)
cs.store(
    group="habitat",
    name="environment",
    node=EnvironmentConfig,
)
cs.store(
    group="habitat",
    name="task",
    node=TaskConfig,
)
cs.store(
    group="habitat.task.actions.stop",
    name="stop",
    node=StopActionConfig,
)
cs.store(
    group="habitat.task.actions.move_forward",
    name="move_forward",
    node=MoveForwardActionConfig,
)
cs.store(
    group="habitat.task.actions.turn_left",
    name="turn_left",
    node=TurnLeftActionConfig,
)
cs.store(
    group="habitat.task.actions.turn_right",
    name="turn_right",
    node=TurnRightActionConfig,
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
