from dataclasses import dataclass, field
from typing import Optional, List, Any

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class IteratorOptions:
    cycle: bool = True
    shuffle: bool = True
    group_by_scene: bool = True
    num_episode_sample: int = -1
    max_scene_repeat_episodes: int = -1
    max_scene_repeat_steps: int = int(1e4)
    step_repetition_range: float = 0.2


@dataclass
class Environment:
    max_episode_steps: int = 1000
    max_episode_seconds: int = 10000000
    iterator_options: IteratorOptions = IteratorOptions()


# -----------------------------------------------------------------------------
# # Actions
# -----------------------------------------------------------------------------
@dataclass
class Action:
    type: str = MISSING


@dataclass
class StopAction(Action):
    type: str = "StopAction"


@dataclass
class EmptyAction(Action):
    type: str = "EmptyAction"


# -----------------------------------------------------------------------------
# # NAVIGATION actions
# -----------------------------------------------------------------------------
@dataclass
class MoveForwardAction(Action):
    type: str = "MoveForwardAction"


@dataclass
class TurnLeftAction(Action):
    type: str = "TurnLeftAction"


@dataclass
class TurnRightAction(Action):
    type: str = "TurnRightAction"


@dataclass
class LookUpAction(Action):
    type: str = "LookUpAction"


@dataclass
class LookDownAction(Action):
    type: str = "LookDownAction"


@dataclass
class TeleportAction(Action):
    type: str = "TeleportAction"


@dataclass
class VelocityControlAction(Action):
    type: str = "VelocityAction"
    # meters/sec
    lin_vel_range: list[float] = field(default_factory=lambda: [0.0, 0.25])
    # deg/sec
    ang_vel_range: list[float] = field(default_factory=lambda: [-10.0, 10.0])
    min_abs_lin_speed: float = 0.025  # meters/sec
    min_abs_ang_speed: float = 1.0  # # deg/sec
    time_step: float = 1.0  # seconds


# -----------------------------------------------------------------------------
# # REARRANGE actions
# -----------------------------------------------------------------------------
@dataclass
class ArmAction(Action):
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
class BaseVelocityAction(Action):
    type: str = "BaseVelAction"
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    end_on_stop: bool = False
    allow_back: bool = True
    min_abs_lin_speed: float = 1.0
    min_abs_ang_speed: float = 1.0


@dataclass
class RearrangeStopAction(Action):
    type: str = "RearrangeStopAction"


@dataclass
class OracleNavAction(Action):
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
class AnswerAction(Action):
    type: str = "AnswerAction"


# -----------------------------------------------------------------------------
# # TASK_SENSORS
# -----------------------------------------------------------------------------
@dataclass
class Sensor:
    type: str = MISSING


@dataclass
class PointGoalSensor(Sensor):
    type: str = "PointGoalSensor"
    goal_format: str = "POLAR"
    dimensionality: int = 2


@dataclass
class PointGoalWithGPSCompassSensor(PointGoalSensor):
    type: str = "PointGoalWithGPSCompassSensor"


class ObjectGoalSensor(Sensor):
    type: str = "ObjectGoalSensor"
    goal_spec: str = "TASK_CATEGORY_ID"
    goal_spec_max_val: int = 50


@dataclass
class ImageGoalSensor(Sensor):
    type: str = "ImageGoalSensor"


@dataclass
class InstanceImageGoalSensor(Sensor):
    type: str = "InstanceImageGoalSensor"


@dataclass
class InstanceImageGoalHFOVSensor(Sensor):
    type: str = "InstanceImageGoalHFOVSensor"


@dataclass
class HeadingSensor(Sensor):
    type: str = "HeadingSensor"


@dataclass
class CompassSensor(Sensor):
    type: str = "CompassSensor"


@dataclass
class GPSSensorSensor(Sensor):
    type: str = "GPSSensor"
    dimensionality: int = 2


@dataclass
class ProximitySensor(Sensor):
    type: str = "ProximitySensor"
    max_detection_radius: float = 2.0


@dataclass
class JointSensor(Sensor):
    type: str = "JointSensor"
    dimensionality: int = 7


@dataclass
class EEPositionSensor(Sensor):
    type: str = "EEPositionSensor"


@dataclass
class IsHoldingSensor(Sensor):
    type: str = "IsHoldingSensor"


@dataclass
class RelativeRestingPositionSensor(Sensor):
    type: str = "RelativeRestingPositionSensor"


@dataclass
class JointVelocitySensor(Sensor):
    type: str = "JointVelocitySensor"
    dimensionality: int = 7


@dataclass
class OracleNavigationActionSensor(Sensor):
    type: str = "OracleNavigationActionSensor"


@dataclass
class RestingPositionSensor(Sensor):
    type: str = "RestingPositionSensor"


@dataclass
class ArtJointSensor(Sensor):
    type: str = "ArtJointSensor"


@dataclass
class NavGoalSensor(Sensor):
    type: str = "NavGoalSensor"


@dataclass
class ArtJointSensorNoVelSensor(Sensor):
    type: str = "ArtJointSensorNoVel"  # TODO: add "Sensor" suffix


@dataclass
class MarkerRelPosSensor(Sensor):
    type: str = "MarkerRelPosSensor"


@dataclass
class TargetStartSensor(Sensor):
    type: str = "TargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class TargetCurrentSensor(Sensor):
    type: str = "TargetCurrentSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class GoalSensor(Sensor):
    type: str = "GoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class TargetOrGoalStartPointGoalSensor(Sensor):
    type: str = "TargetOrGoalStartPointGoalSensor"


@dataclass
class GlobalPredicatesSensor(Sensor):
    type: str = "GlobalPredicatesSensor"


@dataclass
class TargetStartGpsCompassSensor(Sensor):
    type: str = "TargetStartGpsCompassSensor"


@dataclass
class TargetGoalGpsCompassSensor(Sensor):
    type: str = "TargetGoalGpsCompassSensor"


@dataclass
class NavToSkillSensor(Sensor):
    type: str = "NavToSkillSensor"
    num_skills: int = 8


@dataclass
class AbsTargetStartSensor(Sensor):
    type: str = "AbsTargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class AbsGoalSensor(Sensor):
    type: str = "AbsGoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class DistToNavGoalSensor(Sensor):
    type: str = "DistToNavGoalSensor"


@dataclass
class LocalizationSensor(Sensor):
    type: str = "LocalizationSensor"


@dataclass
class QuestionSensor(Sensor):
    type: str = "QuestionSensor"


@dataclass
class InstructionSensor(Sensor):
    type: str = "InstructionSensor"
    instruction_sensor_uuid: str = "instruction"


# -----------------------------------------------------------------------------
# Measurements
# -----------------------------------------------------------------------------
@dataclass
class Measurement:
    type: str = MISSING


@dataclass
class SuccessMeasurement(Measurement):
    type: str = "Success"
    success_distance: float = 0.2


@dataclass
class SPLMeasurement(Measurement):
    type: str = "SPL"


@dataclass
class SoftSPLMeasurement(Measurement):
    type: str = "SoftSPL"


@dataclass
class FogOfWarConfig:
    draw: bool = True
    visibility_dist: float = 5.0
    fov: int = 90


@dataclass
class TopDownMapMeasurement(Measurement):
    type: str = "TopDownMap"
    max_episode_steps: int = Environment.max_episode_steps
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
class CollisionsMeasurement(Measurement):
    type: str = "Collisions"


@dataclass
class RobotForceMeasurement(Measurement):
    type: str = "RobotForce"
    min_force: float = 20.0


@dataclass
class ForceTerminateMeasurement(Measurement):
    type: str = "ForceTerminate"
    max_accum_force: float = -1.0
    max_instant_force: float = -1.0


@dataclass
class RobotCollisionsMeasurement(Measurement):
    type: str = "RobotCollisions"


@dataclass
class ObjectToGoalDistanceMeasurement(Measurement):
    type: str = "ObjectToGoalDistance"


@dataclass
class EndEffectorToObjectDistanceMeasurement(Measurement):
    type: str = "EndEffectorToObjectDistance"


@dataclass
class EndEffectorToRestDistanceMeasurement(Measurement):
    type: str = "EndEffectorToRestDistance"


@dataclass
class ArtObjAtDesiredStateMeasurement(Measurement):
    type: str = "ArtObjAtDesiredState"
    use_absolute_distance: bool = True
    success_dist_threshold: float = 0.05


@dataclass
class GfxReplayMeasureMeasurement(Measurement):
    type: str = "GfxReplayMeasure"


@dataclass
class EndEffectorDistToMarkerMeasurement(Measurement):
    type: str = "EndEffectorDistToMarker"


@dataclass
class ArtObjStateMeasurement(Measurement):
    type: str = "ArtObjState"


@dataclass
class ArtObjSuccessMeasurement(Measurement):
    type: str = "ArtObjSuccess"
    rest_dist_threshold: float = 0.15


@dataclass
class ArtObjRewardMeasurement(Measurement):
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
class RotDistToGoalMeasurement(Measurement):
    type: str = "RotDistToGoal"


@dataclass
class DistToGoalMeasurement(Measurement):
    type: str = "DistToGoal"


@dataclass
class BadCalledTerminateMeasurement(Measurement):
    type: str = "BadCalledTerminate"
    bad_term_pen: float = 0.0
    decay_bad_term: bool = False


@dataclass
class NavToPosSuccMeasurement(Measurement):
    type: str = "NavToPosSucc"
    success_distance: float = 0.2


@dataclass
class NavToObjRewardMeasurement(Measurement):
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
class NavToObjSuccessMeasurement(Measurement):
    type: str = "NavToObjSuccess"
    must_look_at_targ: bool = True
    must_call_stop: bool = True
    # distance in radians.
    success_angle_dist: float = 0.15
    heuristic_stop: bool = False


@dataclass
class RearrangeReachRewardMeasurement(Measurement):
    type: str = "RearrangeReachReward"
    scale: float = 1.0
    diff_reward: bool = True
    sparse_reward: bool = False


@dataclass
class RearrangeReachSuccessMeasurement(Measurement):
    type: str = "RearrangeReachSuccess"
    succ_thresh: float = 0.2


@dataclass
class NumStepsMeasurement(Measurement):
    type: str = "NumStepsMeasure"


@dataclass
class DidPickObjectMeasurement(Measurement):
    type: str = "DidPickObjectMeasure"


@dataclass
class DidViolateHoldConstraintMeasurement(Measurement):
    type: str = "DidViolateHoldConstraintMeasure"


@dataclass
class MoveObjectsRewardMeasurement(Measurement):
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
class RearrangePickRewardMeasurement(Measurement):
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
class RearrangeReachSuccessMeasurement(Measurement):
    type: str = "RearrangePickSuccess"
    ee_resting_success_threshold: float = 0.15


@dataclass
class ObjAtGoalMeasurement(Measurement):
    type: str = "ObjAtGoal"
    succ_thresh: float = 0.15


@dataclass
class PlaceRewardMeasurement(Measurement):
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
class PlaceSuccessMeasurement(Measurement):
    type: str = "PlaceSuccess"
    ee_resting_success_threshold: float = 0.15


@dataclass
class CompositeNodeIdxMeasurement(Measurement):
    type: str = "CompositeNodeIdx"


@dataclass
class CompositeStageGoalsMeasurement(Measurement):
    type: str = "CompositeStageGoals"


@dataclass
class CompositeSuccessMeasurement(Measurement):
    type: str = "CompositeSuccess"
    must_call_stop: bool = True


@dataclass
class CompositeRewardMeasurement(Measurement):
    type: str = "CompositeReward"
    must_call_stop: bool = True
    success_reward: float = 10.0


@dataclass
class DoesWantTerminateMeasurement(Measurement):
    type: str = "DoesWantTerminate"


@dataclass
class CompositeBadCalledTerminateMeasurement(Measurement):
    type: str = "CompositeBadCalledTerminate"


@dataclass
class CorrectAnswerMeasurement(Measurement):
    type: str = "CorrectAnswer"


@dataclass
class EpisodeInfoMeasurement(Measurement):
    type: str = "EpisodeInfo"


@dataclass
class DistanceToGoalMeasurement(Measurement):
    type: str = "DistanceToGoal"
    distance_to: str = "POINT"


@dataclass
class DistanceToGoalRewardMeasurement(Measurement):
    type: str = "DistanceToGoalReward"


@dataclass
class AnswerAccuracyMeasurement(Measurement):
    type: str = "AnswerAccuracy"


@dataclass
class Task:
    reward_measure: Optional[str] = None
    success_measure: Optional[str] = None
    success_reward: float = 2.5
    slack_reward: float = -0.01
    end_on_success: bool = False
    # NAVIGATION task
    type: str = "Nav-v0"
    sensors: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    goal_sensor_uuid: str = "pointgoal"
    possible_actions: list[str] = field(
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
    desired_resting_position: list[float] = field(
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
    filter_nav_to_tasks: list = field(default_factory=list)
    actions: Any = MISSING


@dataclass
class SimulatorSensor:
    type: str = MISSING
    height: int = 480
    width: int = 640
    position: list[float] = field(default_factory=lambda: [0.0, 1.25, 0.0])
    # Euler's angles:
    orientation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class SimulatorCameraSensor(SimulatorSensor):
    hfov: int = 90  # horizontal field of view in degrees
    sensor_subtype: str = "PINHOLE"


@dataclass
class SimulatorDepthSensor(SimulatorSensor):
    min_depth: float = 0.0
    max_depth: float = 10.0
    normalize_depth: bool = True


@dataclass
class HabitatSimRGBSensor(SimulatorCameraSensor):
    type: str = "HabitatSimRGBSensor"


@dataclass
class HabitatSimDepthSensor(SimulatorDepthSensor):
    type: str = "HabitatSimDepthSensor"


@dataclass
class HabitatSimSemanticSensor(SimulatorSensor):
    type: str = "HabitatSimSemanticSensor"


@dataclass
class HabitatSimEquirectangularRGBSensor(SimulatorSensor):
    type: str = "HabitatSimEquirectangularRGBSensor"


@dataclass
class HabitatSimEquirectangularDepthSensor(SimulatorDepthSensor):
    type: str = "HabitatSimEquirectangularDepthSensor"


@dataclass
class HabitatSimEquirectangularSemanticSensor(SimulatorSensor):
    type: str = "HabitatSimEquirectangularSemanticSensor"


@dataclass
class SimulatorFisheyeSensor(SimulatorSensor):
    type: str = "HabitatSimFisheyeSensor"
    height: int = SimulatorSensor.width
    # The default value (alpha, xi) is set to match the lens  "GoPro" found in
    # Table 3 of this paper: Vladyslav Usenko, Nikolaus Demmel and
    # Daniel Cremers: The Double Sphere Camera Model,
    # The International Conference on 3D Vision (3DV), 2018
    # You can find the intrinsic parameters for the other lenses
    # in the same table as well.
    xi: float = -0.27
    alpha: float = 0.57
    focal_length: list[float] = field(default_factory=lambda: [364.84, 364.86])
    # Place camera at center of screen
    # Can be specified, otherwise is calculated automatically.
    # principal_point_offset defaults to (h/2,w/2)
    principal_point_offset: Optional[list[float]] = None
    sensor_model_type: str = "DOUBLE_SPHERE"


@dataclass
class HabitatSimFisheyeRGBSensor(SimulatorFisheyeSensor):
    type: str = "HabitatSimFisheyeRGBSensor"


@dataclass
class SimulatorFisheyeDepthSensor(SimulatorFisheyeSensor):
    type: str = "HabitatSimFisheyeDepthSensor"
    min_depth: float = SimulatorDepthSensor.min_depth
    max_depth: float = SimulatorDepthSensor.max_depth
    normalize_depth: bool = SimulatorDepthSensor.normalize_depth


@dataclass
class HabitatSimFisheyeSemanticSensor(SimulatorFisheyeSensor):
    type: str = "HabitatSimFisheyeSemanticSensor"


@dataclass
class HeadRGBSensor(HabitatSimRGBSensor):
    uuid: str = "robot_head_rgb"


@dataclass
class HeadDepthSensor(HabitatSimDepthSensor):
    uuid: str = "robot_head_depth"


@dataclass
class ArmRGBSensor(HabitatSimRGBSensor):
    uuid: str = "robot_arm_rgb"


@dataclass
class ArmDepthSensor(HabitatSimDepthSensor):
    uuid: str = "robot_arm_depth"


@dataclass
class ThirdRGBSensor(HabitatSimRGBSensor):
    uuid: str = "robot_third_rgb"


@dataclass
class ThirdDepthSensor(HabitatSimDepthSensor):
    uuid: str = "robot_third_depth"  # TODO: robot_third_rgb on the main branch
                                     #  check if it won't cause any errors


@dataclass
class Agent:
    height: float = 1.5
    radius: float = 0.1
    sensors: list[str] = field(default_factory=lambda: ["rgb_sensor"])
    is_set_start_state: bool = False
    start_position: list[float] = field(default_factory=lambda: [0, 0, 0])
    start_rotation: list[float] = field(default_factory=lambda: [0, 0, 0, 1])
    joint_start_noise: float = 0.0
    robot_urdf: str = "data/robots/hab_fetch/robots/hab_fetch.urdf"
    robot_type: str = "FetchRobot"
    ik_arm_urdf: str = "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"


@dataclass
class HabitatSimV0:
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
class Simulator:
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
    additional_object_paths: list = field(default_factory=list)
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
    agents: list[str] = field(default_factory=lambda: ["agent_0"])
    agent_0: Agent = Agent()
    rgb_sensor: HabitatSimRGBSensor = HabitatSimRGBSensor()
    depth_sensor: HabitatSimDepthSensor = HabitatSimDepthSensor()
    habitat_sim_v0: HabitatSimV0 = HabitatSimV0()


@dataclass
class Pyrobot:
    # types of robots supported:
    robots: list[str] = field(default_factory=lambda: ["locobot"])
    robot: str = "locobot"
    sensors: list[str] = field(
        default_factory=lambda: [
            "rgb_sensor",
            "depth_sensor",
            "bump_sensor"
        ]
    )
    base_controller: str = "proportional"
    base_planner: str = "none"


@dataclass
class PyrobotVisualSensor:
    type: str = MISSING
    height: int = 480
    width: int = 640


@dataclass
class PyrobotRGBSensor(PyrobotVisualSensor):
    type: str = "PyRobotRGBSensor"
    center_crop: bool = False


@dataclass
class PyrobotDepthSensor(PyrobotVisualSensor):
    type: str = "PyRobotDepthSensor"
    min_depth: float = 0.0
    max_depth: float = 5.0
    normalize_depth: bool = True
    center_crop: bool = False


@dataclass
class PyrobotBumpSensor:
    type: str = "PyRobotBumpSensor"


@dataclass
class Locobot:
    actions: list[str] = field(
        default_factory=lambda: [
            "base_actions",
            "camera_actions"
        ]
    )
    base_actions: list[str] = field(
        default_factory=lambda: [
            "go_to_relative",
            "go_to_absolute"
        ]
    )
    camera_actions: list[str] = field(
        default_factory=lambda: [
            "set_pan",
            "set_tilt",
            "set_pan_tilt"
        ]
    )


@dataclass
class Dataset:
    type: str = "PointNav-v1"
    split: str = "train"
    scenes_dir: str = "data/scene_datasets"
    content_scenes: list[str] = field(default_factory=lambda: ["*"])
    data_path: str = "data/datasets/pointnav/" \
                "habitat-test-scenes/v1/{split}/{split}.json.gz"


@dataclass
class Gym:
    auto_name: str = ""
    obs_keys: Optional[list[str]] = None
    action_keys: Optional[list[str]] = None
    achieved_goal_keys: list = field(default_factory=list)
    desired_goal_keys: list[str] = field(default_factory=list)


@dataclass
class Config:
    seed: int = 100
    # GymHabitatEnv works for all Habitat tasks, including Navigation and
    # Rearrange. To use a gym environment from the registry, use the
    # GymRegistryEnv. Any other environment needs to be created and registered.
    env_task: str = "GymHabitatEnv"
    # The dependencies for launching the GymRegistryEnv environments.
    # Modules listed here will be imported prior to making the environment with
    # gym.make()
    env_task_gym_dependencies: list = field(default_factory=list)
    # The key of the gym environment in the registry to use in GymRegistryEnv
    # for example: `Cartpole-v0`
    env_task_gym_id: str = ""
    environment: Environment = Environment()
    simulator: Simulator = Simulator()
    task: Task = Task()
    dataset: Dataset = Dataset()


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(
    group="habitat",
    name="config",
    node=Config
)
cs.store(
    group="habitat",
    name="environment",
    node=Environment,
)
cs.store(
    group="habitat",
    name="task",
    node=Task,
)
cs.store(
    group="habitat.task.actions.stop",
    name="stop",
    node=StopAction,
)
cs.store(
    group="habitat.task.actions.move_forward",
    name="move_forward",
    node=MoveForwardAction,
)
cs.store(
    group="habitat.task.actions.turn_left",
    name="turn_left",
    node=TurnLeftAction,
)
cs.store(
    group="habitat.task.actions.turn_right",
    name="turn_right",
    node=TurnRightAction,
)
