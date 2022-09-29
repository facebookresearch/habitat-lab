#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
from typing import List, Optional, Union

import yacs.config

from habitat.core.logging import logger


# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

_HABITAT_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
# This is equivalent to doing osp.dirname(osp.abspath(__file__))
# in editable install, this is pwd/habitat-lab/habitat/config


CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.habitat = CN()
_C.habitat.seed = 100
# -----------------------------------------------------------------------------
# environment
# -----------------------------------------------------------------------------
_C.habitat.environment = CN()
_C.habitat.environment.max_episode_steps = 1000
_C.habitat.environment.max_episode_seconds = 10000000
_C.habitat.environment.iterator_options = CN()
_C.habitat.environment.iterator_options.cycle = True
_C.habitat.environment.iterator_options.shuffle = True
_C.habitat.environment.iterator_options.group_by_scene = True
_C.habitat.environment.iterator_options.num_episode_sample = -1
_C.habitat.environment.iterator_options.max_scene_repeat_episodes = -1
_C.habitat.environment.iterator_options.max_scene_repeat_steps = int(1e4)
_C.habitat.environment.iterator_options.step_repetition_range = 0.2
# -----------------------------------------------------------------------------
# task
# -----------------------------------------------------------------------------
_C.habitat.task = CN()
_C.habitat.task.reward_measure = None
_C.habitat.task.success_measure = None
_C.habitat.task.success_reward = 2.5
_C.habitat.task.slack_reward = -0.01
_C.habitat.task.end_on_success = False
# -----------------------------------------------------------------------------
# # NAVIGATION task
# -----------------------------------------------------------------------------
_C.habitat.task.type = "Nav-v0"
_C.habitat.task.sensors = []
_C.habitat.task.measurements = []
_C.habitat.task.goal_sensor_uuid = "pointgoal"
_C.habitat.task.possible_actions = [
    "stop",
    "move_forward",
    "turn_left",
    "turn_right",
]
# -----------------------------------------------------------------------------
# # REARRANGE task
# -----------------------------------------------------------------------------
_C.habitat.task.count_obj_collisions = True
_C.habitat.task.settle_steps = 5
_C.habitat.task.constraint_violation_ends_episode = True
_C.habitat.task.constraint_violation_drops_object = False
_C.habitat.task.force_regenerate = (
    False  # Forced to regenerate the starts even if they are already cached.
)
_C.habitat.task.should_save_to_cache = True  # Saves the generated starts to a cache if they are not already generated.
_C.habitat.task.must_look_at_targ = True
_C.habitat.task.object_in_hand_sample_prob = 0.167
_C.habitat.task.gfx_replay_dir = "data/replays"
_C.habitat.task.render_target = True
_C.habitat.task.ee_sample_factor = 0.2
_C.habitat.task.ee_exclude_region = 0.0
# In radians
_C.habitat.task.base_angle_noise = 0.15
_C.habitat.task.base_noise = 0.05
_C.habitat.task.spawn_region_scale = 0.2
_C.habitat.task.joint_max_impulse = -1.0
_C.habitat.task.desired_resting_position = [0.5, 0.0, 1.0]
_C.habitat.task.use_marker_t = True
_C.habitat.task.cache_robot_init = False
_C.habitat.task.success_state = 0.0
# If true, does not care about navigability or collisions with objects when spawning
# robot
_C.habitat.task.easy_init = False
_C.habitat.task.should_enforce_target_within_reach = False
# -----------------------------------------------------------------------------
# # COMPOSITE task CONFIG
# -----------------------------------------------------------------------------
_C.habitat.task.task_spec_base_path = "tasks/rearrange/pddl/"
_C.habitat.task.task_spec = ""
# PDDL domain params
_C.habitat.task.pddl_domain_def = "replica_cad"
_C.habitat.task.obj_succ_thresh = 0.3
_C.habitat.task.art_succ_thresh = 0.15
_C.habitat.task.robot_at_thresh = 2.0
_C.habitat.task.filter_nav_to_tasks = []
# -----------------------------------------------------------------------------
# # actions
# -----------------------------------------------------------------------------
_C.habitat.task.actions = CN()
_C.habitat.task.actions.stop = CN()
_C.habitat.task.actions.stop.type = "StopAction"
_C.habitat.task.actions.empty = CN()
_C.habitat.task.actions.empty.type = "EmptyAction"
# -----------------------------------------------------------------------------
# # NAVIGATION actions
# -----------------------------------------------------------------------------
_C.habitat.task.actions.move_forward = CN()
_C.habitat.task.actions.move_forward.type = "MoveForwardAction"
_C.habitat.task.actions.turn_left = CN()
_C.habitat.task.actions.turn_left.type = "TurnLeftAction"
_C.habitat.task.actions.turn_right = CN()
_C.habitat.task.actions.turn_right.type = "TurnRightAction"
_C.habitat.task.actions.look_up = CN()
_C.habitat.task.actions.look_up.type = "LookUpAction"
_C.habitat.task.actions.look_down = CN()
_C.habitat.task.actions.look_down.type = "LookDownAction"
_C.habitat.task.actions.teleport = CN()
_C.habitat.task.actions.teleport.type = "TeleportAction"
_C.habitat.task.actions.velocity_control = CN()
_C.habitat.task.actions.velocity_control.type = "VelocityAction"
_C.habitat.task.actions.velocity_control.lin_vel_range = [
    0.0,
    0.25,
]  # meters per sec
_C.habitat.task.actions.velocity_control.ang_vel_range = [
    -10.0,
    10.0,
]  # deg per sec
_C.habitat.task.actions.velocity_control.min_abs_lin_speed = (
    0.025  # meters per sec
)
_C.habitat.task.actions.velocity_control.min_abs_ang_speed = 1.0  # deg per sec
_C.habitat.task.actions.velocity_control.time_step = 1.0  # seconds
# -----------------------------------------------------------------------------
# # REARRANGE actions
# -----------------------------------------------------------------------------
_C.habitat.task.actions.arm_action = CN()
_C.habitat.task.actions.arm_action.type = "ArmAction"
_C.habitat.task.actions.arm_action.arm_controller = "ArmRelPosAction"
_C.habitat.task.actions.arm_action.grip_controller = None
_C.habitat.task.actions.arm_action.arm_joint_dimensionality = 7
_C.habitat.task.actions.arm_action.grasp_thresh_dist = 0.15
_C.habitat.task.actions.arm_action.disable_grip = False
_C.habitat.task.actions.arm_action.delta_pos_limit = 0.0125
_C.habitat.task.actions.arm_action.ee_ctrl_lim = 0.015
_C.habitat.task.actions.arm_action.should_clip = False
_C.habitat.task.actions.arm_action.render_ee_target = False
_C.habitat.task.actions.arm_action.agent = None
_C.habitat.task.actions.base_velocity = CN()
_C.habitat.task.actions.base_velocity.type = "BaseVelAction"
_C.habitat.task.actions.base_velocity.lin_speed = 10.0
_C.habitat.task.actions.base_velocity.ang_speed = 10.0
_C.habitat.task.actions.base_velocity.allow_dyn_slide = True
_C.habitat.task.actions.base_velocity.end_on_stop = False
_C.habitat.task.actions.base_velocity.allow_back = True
_C.habitat.task.actions.base_velocity.min_abs_lin_speed = 1.0
_C.habitat.task.actions.base_velocity.min_abs_ang_speed = 1.0
_C.habitat.task.actions.base_velocity.agent = None
_C.habitat.task.actions.rearrange_stop = CN()
_C.habitat.task.actions.rearrange_stop.type = "RearrangeStopAction"
# -----------------------------------------------------------------------------
# Oracle navigation action
# This action takes as input a discrete ID which refers to an object in the
# PDDL domain. The oracle navigation controller then computes the actions to
# navigate to that desired object.
# -----------------------------------------------------------------------------
_C.habitat.task.actions.oracle_nav_action = CN()
_C.habitat.task.actions.oracle_nav_action.type = "OracleNavAction"
_C.habitat.task.actions.oracle_nav_action.turn_velocity = 1.0
_C.habitat.task.actions.oracle_nav_action.forward_velocity = 1.0
_C.habitat.task.actions.oracle_nav_action.turn_thresh = 0.1
_C.habitat.task.actions.oracle_nav_action.dist_thresh = 0.2
_C.habitat.task.actions.oracle_nav_action.agent = None
_C.habitat.task.actions.oracle_nav_action.lin_speed = 10.0
_C.habitat.task.actions.oracle_nav_action.ang_speed = 10.0
_C.habitat.task.actions.oracle_nav_action.min_abs_lin_speed = 1.0
_C.habitat.task.actions.oracle_nav_action.min_abs_ang_speed = 1.0
_C.habitat.task.actions.oracle_nav_action.allow_dyn_slide = True
_C.habitat.task.actions.oracle_nav_action.end_on_stop = False
_C.habitat.task.actions.oracle_nav_action.allow_back = True
# -----------------------------------------------------------------------------
# # TASK_SENSORS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.pointgoal_sensor = CN()
_C.habitat.task.pointgoal_sensor.type = "PointGoalSensor"
_C.habitat.task.pointgoal_sensor.goal_format = "POLAR"
_C.habitat.task.pointgoal_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.pointgoal_with_gps_compass_sensor = (
    _C.habitat.task.pointgoal_sensor.clone()
)
_C.habitat.task.pointgoal_with_gps_compass_sensor.type = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# OBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.OBJECTgoal_sensor = CN()
_C.habitat.task.OBJECTgoal_sensor.type = "ObjectGoalSensor"
_C.habitat.task.OBJECTgoal_sensor.goal_spec = "TASK_CATEGORY_ID"
_C.habitat.task.OBJECTgoal_sensor.goal_spec_max_val = 50
# -----------------------------------------------------------------------------
# IMAGEGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.imagegoal_sensor = CN()
_C.habitat.task.imagegoal_sensor.type = "ImageGoalSensor"
# -----------------------------------------------------------------------------
# INSTANCE IMAGEGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.instance_imagegoal_sensor = CN()
_C.habitat.task.instance_imagegoal_sensor.type = "InstanceImageGoalSensor"
# -----------------------------------------------------------------------------
# INSTANCE IMAGEGOAL HFOV SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.instance_imagegoal_hfov_sensor = CN()
_C.habitat.task.instance_imagegoal_hfov_sensor.type = (
    "InstanceImageGoalHFOVSensor"
)
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.heading_sensor = CN()
_C.habitat.task.heading_sensor.type = "HeadingSensor"
# -----------------------------------------------------------------------------
# COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.compass_sensor = CN()
_C.habitat.task.compass_sensor.type = "CompassSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.gps_sensor = CN()
_C.habitat.task.gps_sensor.type = "GPSSensor"
_C.habitat.task.gps_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.proximity_sensor = CN()
_C.habitat.task.proximity_sensor.type = "ProximitySensor"
_C.habitat.task.proximity_sensor.max_detection_radius = 2.0
# -----------------------------------------------------------------------------
# JOINT SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.joint_sensor = CN()
_C.habitat.task.joint_sensor.type = "JointSensor"
_C.habitat.task.joint_sensor.dimensionality = 7
# -----------------------------------------------------------------------------
# END EFFECTOR POSITION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.end_effector_sensor = CN()
_C.habitat.task.end_effector_sensor.type = "EEPositionSensor"
# -----------------------------------------------------------------------------
# IS HOLDING SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.is_holding_sensor = CN()
_C.habitat.task.is_holding_sensor.type = "IsHoldingSensor"
# -----------------------------------------------------------------------------
# RELATIVE RESTING POSISITON SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.relative_resting_pos_sensor = CN()
_C.habitat.task.relative_resting_pos_sensor.type = (
    "RelativeRestingPositionSensor"
)
# -----------------------------------------------------------------------------
# JOINT VELOCITY SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.joint_velocity_sensor = CN()
_C.habitat.task.joint_velocity_sensor.type = "JointVelocitySensor"
_C.habitat.task.joint_velocity_sensor.dimensionality = 7
# -----------------------------------------------------------------------------
# ORACLE NAVIGATION ACTION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.oracle_nav_action_SENSOR = CN()
_C.habitat.task.oracle_nav_action_SENSOR.type = "OracleNavigationActionSensor"
# -----------------------------------------------------------------------------
# RESTING POSITION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.resting_pos_sensor = CN()
_C.habitat.task.resting_pos_sensor.type = "RestingPositionSensor"
# -----------------------------------------------------------------------------
# ART JOINT SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.ART_joint_sensor = CN()
_C.habitat.task.ART_joint_sensor.type = "ArtJointSensor"
# -----------------------------------------------------------------------------
# NAV GOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.NAV_goal_sensor = CN()
_C.habitat.task.NAV_goal_sensor.type = "NavGoalSensor"
# -----------------------------------------------------------------------------
# ART JOINT NO VELOCITY SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.ART_joint_sensor_NO_VEL = CN()
_C.habitat.task.ART_joint_sensor_NO_VEL.type = "ArtJointSensorNoVel"
# -----------------------------------------------------------------------------
# MARKER RELATIVE POSISITON SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.marker_rel_pos_sensor = CN()
_C.habitat.task.marker_rel_pos_sensor.type = "MarkerRelPosSensor"
# -----------------------------------------------------------------------------
# TARGET START SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.target_start_sensor = CN()
_C.habitat.task.target_start_sensor.type = "TargetStartSensor"
_C.habitat.task.target_start_sensor.goal_format = "CARTESIAN"
_C.habitat.task.target_start_sensor.dimensionality = 3
# -----------------------------------------------------------------------------
# OBJECT SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.object_sensor = CN()
_C.habitat.task.object_sensor.type = "TargetCurrentSensor"
_C.habitat.task.object_sensor.goal_format = "CARTESIAN"
_C.habitat.task.object_sensor.dimensionality = 3

# -----------------------------------------------------------------------------
# GOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.goal_sensor = CN()
_C.habitat.task.goal_sensor.type = "GoalSensor"
_C.habitat.task.goal_sensor.goal_format = "CARTESIAN"
_C.habitat.task.goal_sensor.dimensionality = 3
# -----------------------------------------------------------------------------
# TARGET START OR GOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.target_start_point_goal_sensor = CN()
_C.habitat.task.target_start_point_goal_sensor.type = (
    "TargetOrGoalStartPointGoalSensor"
)
# -----------------------------------------------------------------------------
# COMPOSITE SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.global_predicate_sensor = CN()
_C.habitat.task.global_predicate_sensor.type = "GlobalPredicatesSensor"
# -----------------------------------------------------------------------------
# TARGET START GPS/COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.target_start_gps_compass_sensor = CN()
_C.habitat.task.target_start_gps_compass_sensor.type = (
    "TargetStartGpsCompassSensor"
)
# -----------------------------------------------------------------------------
# TARGET GOAL GPS/COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.target_goal_gps_compass_sensor = CN()
_C.habitat.task.target_goal_gps_compass_sensor.type = (
    "TargetGoalGpsCompassSensor"
)
# -----------------------------------------------------------------------------
# NAV TO SKILL ID SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.nav_to_skill_sensor = CN()
_C.habitat.task.nav_to_skill_sensor.type = "NavToSkillSensor"
_C.habitat.task.nav_to_skill_sensor.num_skills = 8
# -----------------------------------------------------------------------------
# ABSOLUTE TARGET START SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.abs_target_start_sensor = CN()
_C.habitat.task.abs_target_start_sensor.type = "AbsTargetStartSensor"
_C.habitat.task.abs_target_start_sensor.goal_format = "CARTESIAN"
_C.habitat.task.abs_target_start_sensor.dimensionality = 3
# -----------------------------------------------------------------------------
# ABSOLUTE GOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.ABS_goal_sensor = CN()
_C.habitat.task.ABS_goal_sensor.type = "AbsGoalSensor"
_C.habitat.task.ABS_goal_sensor.goal_format = "CARTESIAN"
_C.habitat.task.ABS_goal_sensor.dimensionality = 3
# -----------------------------------------------------------------------------
# DISTANCE TO NAVIGATION GOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.dist_to_nav_goal = CN()
_C.habitat.task.dist_to_nav_goal.type = "DistToNavGoalSensor"
# -----------------------------------------------------------------------------
# LOCALIZATION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.localization_sensor = CN()
_C.habitat.task.localization_sensor.type = "LocalizationSensor"
# -----------------------------------------------------------------------------
# SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.success = CN()
_C.habitat.task.success.type = "Success"
_C.habitat.task.success.success_distance = 0.2
# -----------------------------------------------------------------------------
# SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.spl = CN()
_C.habitat.task.spl.type = "SPL"
# -----------------------------------------------------------------------------
# SOFT-SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.soft_spl = CN()
_C.habitat.task.soft_spl.type = "SoftSPL"
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.top_down_map = CN()
_C.habitat.task.top_down_map.type = "TopDownMap"
_C.habitat.task.top_down_map.max_episode_steps = (
    _C.habitat.environment.max_episode_steps
)
_C.habitat.task.top_down_map.map_padding = 3
_C.habitat.task.top_down_map.map_resolution = 1024
_C.habitat.task.top_down_map.draw_source = True
_C.habitat.task.top_down_map.draw_border = True
_C.habitat.task.top_down_map.draw_shortest_path = True
_C.habitat.task.top_down_map.fog_of_war = CN()
_C.habitat.task.top_down_map.fog_of_war.draw = True
_C.habitat.task.top_down_map.fog_of_war.visibility_dist = 5.0
_C.habitat.task.top_down_map.fog_of_war.fov = 90
_C.habitat.task.top_down_map.draw_view_points = True
_C.habitat.task.top_down_map.draw_goal_positions = True
# Axes aligned bounding boxes
_C.habitat.task.top_down_map.draw_goal_aabbs = True
# -----------------------------------------------------------------------------
# COLLISIONSMEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.collisions = CN()
_C.habitat.task.collisions.type = "Collisions"
# -----------------------------------------------------------------------------
# GENERAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.robot_force = CN()
_C.habitat.task.robot_force.type = "RobotForce"
_C.habitat.task.robot_force.min_force = 20.0

_C.habitat.task.force_terminate = CN()
_C.habitat.task.force_terminate.type = "ForceTerminate"
_C.habitat.task.force_terminate.max_accum_force = -1.0
_C.habitat.task.force_terminate.max_instant_force = -1.0

_C.habitat.task.robot_colls = CN()
_C.habitat.task.robot_colls.type = "RobotCollisions"
_C.habitat.task.object_to_goal_distance = CN()
_C.habitat.task.object_to_goal_distance.type = "ObjectToGoalDistance"
_C.habitat.task.end_effector_to_object_distance = CN()
_C.habitat.task.end_effector_to_object_distance.type = (
    "EndEffectorToObjectDistance"
)
_C.habitat.task.end_effector_to_rest_distance = CN()
_C.habitat.task.end_effector_to_rest_distance.type = (
    "EndEffectorToRestDistance"
)

_C.habitat.task.art_obj_at_desired_state = CN()
_C.habitat.task.art_obj_at_desired_state.type = "ArtObjAtDesiredState"
_C.habitat.task.art_obj_at_desired_state.use_absolute_distance = True
_C.habitat.task.art_obj_at_desired_state.success_dist_threshold = 0.05

_C.habitat.task.gfx_replay_measure = CN()
_C.habitat.task.gfx_replay_measure.type = "GfxReplayMeasure"

_C.habitat.task.ee_dist_to_marker = CN()
_C.habitat.task.ee_dist_to_marker.type = "EndEffectorDistToMarker"
_C.habitat.task.art_obj_state = CN()
_C.habitat.task.art_obj_state.type = "ArtObjState"
_C.habitat.task.art_obj_success = CN()
_C.habitat.task.art_obj_success.type = "ArtObjSuccess"
_C.habitat.task.art_obj_success.rest_dist_threshold = 0.15

_C.habitat.task.art_obj_reward = CN()
_C.habitat.task.art_obj_reward.type = "ArtObjReward"
_C.habitat.task.art_obj_reward.dist_reward = 1.0
_C.habitat.task.art_obj_reward.wrong_grasp_end = False
_C.habitat.task.art_obj_reward.wrong_grasp_pen = 5.0
_C.habitat.task.art_obj_reward.art_dist_reward = 10.0
_C.habitat.task.art_obj_reward.ee_dist_reward = 10.0
_C.habitat.task.art_obj_reward.marker_dist_reward = 0.0
_C.habitat.task.art_obj_reward.art_at_desired_state_reward = 5.0
_C.habitat.task.art_obj_reward.grasp_reward = 0.0
# General Rearrange Reward config
_C.habitat.task.art_obj_reward.constraint_violate_pen = 10.0
_C.habitat.task.art_obj_reward.force_pen = 0.0
_C.habitat.task.art_obj_reward.max_force_pen = 1.0
_C.habitat.task.art_obj_reward.force_end_pen = 10.0
# -----------------------------------------------------------------------------
# NAVIGATION MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.rot_dist_to_goal = CN()
_C.habitat.task.rot_dist_to_goal.type = "RotDistToGoal"
_C.habitat.task.dist_to_goal = CN()
_C.habitat.task.dist_to_goal.type = "DistToGoal"
_C.habitat.task.bad_called_terminate = CN()
_C.habitat.task.bad_called_terminate.type = "BadCalledTerminate"
_C.habitat.task.bad_called_terminate.bad_term_pen = 0.0
_C.habitat.task.bad_called_terminate.decay_bad_term = False
_C.habitat.task.nav_to_pos_succ = CN()
_C.habitat.task.nav_to_pos_succ.type = "NavToPosSucc"
_C.habitat.task.nav_to_pos_succ.success_distance = 0.2
# -----------------------------------------------------------------------------
# REARRANGE NAVIGATION MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.rearrange_nav_to_obj_reward = CN()
_C.habitat.task.rearrange_nav_to_obj_reward.type = "NavToObjReward"
# Reward the agent for facing the object?
_C.habitat.task.rearrange_nav_to_obj_reward.should_reward_turn = True
# What distance do we start giving the reward for facing the object?
_C.habitat.task.rearrange_nav_to_obj_reward.turn_reward_dist = 0.1
# Multiplier on the angle distance to the goal.
_C.habitat.task.rearrange_nav_to_obj_reward.angle_dist_reward = 1.0
_C.habitat.task.rearrange_nav_to_obj_reward.dist_reward = 10.0
_C.habitat.task.rearrange_nav_to_obj_reward.constraint_violate_pen = 10.0
_C.habitat.task.rearrange_nav_to_obj_reward.force_pen = 0.0
_C.habitat.task.rearrange_nav_to_obj_reward.max_force_pen = 1.0
_C.habitat.task.rearrange_nav_to_obj_reward.force_end_pen = 10.0

_C.habitat.task.rearrange_nav_to_obj_success = CN()
_C.habitat.task.rearrange_nav_to_obj_success.type = "NavToObjSuccess"
_C.habitat.task.rearrange_nav_to_obj_success.must_look_at_targ = True
_C.habitat.task.rearrange_nav_to_obj_success.must_call_stop = True
# Distance in radians.
_C.habitat.task.rearrange_nav_to_obj_success.success_angle_dist = 0.15
_C.habitat.task.rearrange_nav_to_obj_success.heuristic_stop = False
# -----------------------------------------------------------------------------
# REARRANGE REACH MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.rearrange_reach_reward = CN()
_C.habitat.task.rearrange_reach_reward.type = "RearrangeReachReward"
_C.habitat.task.rearrange_reach_reward.scale = 1.0
_C.habitat.task.rearrange_reach_reward.diff_reward = True
_C.habitat.task.rearrange_reach_reward.sparse_reward = False

_C.habitat.task.rearrange_reach_success = CN()
_C.habitat.task.rearrange_reach_success.type = "RearrangeReachSuccess"
_C.habitat.task.rearrange_reach_success.succ_thresh = 0.2
# -----------------------------------------------------------------------------
# NUM STEPS MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.num_steps = CN()
_C.habitat.task.num_steps.type = "NumStepsMeasure"
# -----------------------------------------------------------------------------
# DID PICK OBJECT MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.did_pick_object = CN()
_C.habitat.task.did_pick_object.type = "DidPickObjectMeasure"
# -----------------------------------------------------------------------------
# DID VIOLATE HOLD CONSTRAINT MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.did_violate_hold_constraint = CN()
_C.habitat.task.did_violate_hold_constraint.type = (
    "DidViolateHoldConstraintMeasure"
)
# -----------------------------------------------------------------------------
# MOVE OBJECTS REWARD
# -----------------------------------------------------------------------------
_C.habitat.task.move_objects_reward = CN()
_C.habitat.task.move_objects_reward.type = "MoveObjectsReward"
_C.habitat.task.move_objects_reward.pick_reward = 1.0
_C.habitat.task.move_objects_reward.success_dist = 0.15
_C.habitat.task.move_objects_reward.single_rearrange_reward = 1.0
_C.habitat.task.move_objects_reward.dist_reward = 1.0
_C.habitat.task.move_objects_reward.constraint_violate_pen = 10.0
_C.habitat.task.move_objects_reward.force_pen = 0.001
_C.habitat.task.move_objects_reward.max_force_pen = 1.0
_C.habitat.task.move_objects_reward.force_end_pen = 10.0
# -----------------------------------------------------------------------------
# PICK MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.pick_reward = CN()
_C.habitat.task.pick_reward.type = "RearrangePickReward"
_C.habitat.task.pick_reward.dist_reward = 20.0
_C.habitat.task.pick_reward.succ_reward = 10.0
_C.habitat.task.pick_reward.pick_reward = 20.0
_C.habitat.task.pick_reward.constraint_violate_pen = 10.0
_C.habitat.task.pick_reward.drop_pen = 5.0
_C.habitat.task.pick_reward.wrong_pick_pen = 5.0
_C.habitat.task.pick_reward.max_accum_force = 5000.0
_C.habitat.task.pick_reward.force_pen = 0.001
_C.habitat.task.pick_reward.max_force_pen = 1.0
_C.habitat.task.pick_reward.force_end_pen = 10.0
_C.habitat.task.pick_reward.use_diff = True
_C.habitat.task.pick_reward.drop_obj_should_end = False
_C.habitat.task.pick_reward.wrong_pick_should_end = False
_C.habitat.task.pick_success = CN()
_C.habitat.task.pick_success.type = "RearrangePickSuccess"
_C.habitat.task.pick_success.ee_resting_success_threshold = 0.15
# -----------------------------------------------------------------------------
# PLACE MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.obj_at_goal = CN()
_C.habitat.task.obj_at_goal.type = "ObjAtGoal"
_C.habitat.task.obj_at_goal.succ_thresh = 0.15

_C.habitat.task.place_reward = CN()
_C.habitat.task.place_reward.type = "PlaceReward"
_C.habitat.task.place_reward.dist_reward = 20.0
_C.habitat.task.place_reward.succ_reward = 10.0
_C.habitat.task.place_reward.place_reward = 20.0
_C.habitat.task.place_reward.drop_pen = 5.0
_C.habitat.task.place_reward.use_diff = True
_C.habitat.task.place_reward.wrong_drop_should_end = False
_C.habitat.task.place_reward.constraint_violate_pen = 10.0
_C.habitat.task.place_reward.force_pen = 0.001
_C.habitat.task.place_reward.max_force_pen = 1.0
_C.habitat.task.place_reward.force_end_pen = 10.0

_C.habitat.task.place_success = CN()
_C.habitat.task.place_success.type = "PlaceSuccess"
_C.habitat.task.place_success.ee_resting_success_threshold = 0.15
# -----------------------------------------------------------------------------
# COMPOSITE MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.composite_node_idx = CN()
_C.habitat.task.composite_node_idx.type = "CompositeNodeIdx"
_C.habitat.task.composite_stage_goals = CN()
_C.habitat.task.composite_stage_goals.type = "CompositeStageGoals"
_C.habitat.task.composite_success = CN()
_C.habitat.task.composite_success.type = "CompositeSuccess"
_C.habitat.task.composite_success.must_call_stop = True
_C.habitat.task.composite_reward = CN()
_C.habitat.task.composite_reward.type = "CompositeReward"
_C.habitat.task.composite_reward.success_reward = 10.0
_C.habitat.task.does_want_terminate = CN()
_C.habitat.task.does_want_terminate.type = "DoesWantTerminate"
_C.habitat.task.composite_bad_called_terminate = CN()
_C.habitat.task.composite_bad_called_terminate.type = (
    "CompositeBadCalledTerminate"
)
# -----------------------------------------------------------------------------
# # EQA task
# -----------------------------------------------------------------------------
_C.habitat.task.actions.answer = CN()
_C.habitat.task.actions.answer.type = "AnswerAction"
# # EQA task QUESTION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.question_sensor = CN()
_C.habitat.task.question_sensor.type = "QuestionSensor"
# -----------------------------------------------------------------------------
# # EQA task correct_answer measure for training
# -----------------------------------------------------------------------------
_C.habitat.task.correct_answer = CN()
_C.habitat.task.correct_answer.type = "CorrectAnswer"
# -----------------------------------------------------------------------------
# # EQA task answer SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.episode_info = CN()
_C.habitat.task.episode_info.type = "EpisodeInfo"
# -----------------------------------------------------------------------------
# # VLN task INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.instruction_sensor = CN()
_C.habitat.task.instruction_sensor.type = "InstructionSensor"
# -----------------------------------------------------------------------------
# # distance_to_goal MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.distance_to_goal = CN()
_C.habitat.task.distance_to_goal.type = "DistanceToGoal"
_C.habitat.task.distance_to_goal.distance_to = "POINT"
# -----------------------------------------------------------------------------
# # distance_to_goal_reward MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.distance_to_goal_reward = CN()
_C.habitat.task.distance_to_goal_reward.type = "DistanceToGoalReward"
# -----------------------------------------------------------------------------
# # answer_accuracy MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.answer_accuracy = CN()
_C.habitat.task.answer_accuracy.type = "AnswerAccuracy"
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.habitat.simulator = CN()
_C.habitat.simulator.type = "Sim-v0"
_C.habitat.simulator.action_space_config = "v0"
_C.habitat.simulator.forward_step_size = 0.25  # in metres
_C.habitat.simulator.create_renderer = False
_C.habitat.simulator.requires_textures = True
_C.habitat.simulator.lag_observations = 0
_C.habitat.simulator.auto_sleep = False
_C.habitat.simulator.step_physics = True
_C.habitat.simulator.update_robot = True
_C.habitat.simulator.concur_render = False
_C.habitat.simulator.needs_markers = (
    True  # If markers should be updated at every step.
)
_C.habitat.simulator.update_robot = (
    True  # If the robot camera positions should be updated at every step.
)
_C.habitat.simulator.scene = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_C.habitat.simulator.scene_dataset = "default"  # the scene dataset to load in the MetaDataMediator. Should contain simulator.scene
_C.habitat.simulator.additional_object_paths = (
    []
)  # a list of directory or config paths to search in addition to the dataset for object configs. Should match the generated episodes for the task.
_C.habitat.simulator.seed = _C.habitat.seed
_C.habitat.simulator.turn_angle = (
    10  # angle to rotate left or right in degrees
)
_C.habitat.simulator.tilt_angle = (
    15  # angle to tilt the camera up or down in degrees
)
_C.habitat.simulator.default_agent_id = 0
_C.habitat.simulator.debug_render = False
_C.habitat.simulator.debug_render_robot = False
_C.habitat.simulator.kinematic_mode = False
# If in render mode a visualization of the rearrangement goal position should
# also be displayed.
_C.habitat.simulator.debug_render_goal = True
_C.habitat.simulator.robot_joint_start_noise = 0.0
# Rearrange Agent Setup
_C.habitat.simulator.ctrl_freq = 120.0
_C.habitat.simulator.ac_freq_ratio = 4
_C.habitat.simulator.load_objs = False
# Rearrange Agent Grasping
_C.habitat.simulator.hold_thresh = 0.09
_C.habitat.simulator.grasp_impulse = 1000.0
# -----------------------------------------------------------------------------
# SIMULATOR SENSORS
# -----------------------------------------------------------------------------
simulator_sensor = CN()
simulator_sensor.height = 480
simulator_sensor.width = 640
simulator_sensor.position = [0, 1.25, 0]
simulator_sensor.orientation = [0.0, 0.0, 0.0]  # Euler's angles

# -----------------------------------------------------------------------------
# CAMERA SENSOR
# -----------------------------------------------------------------------------
camera_sim_sensor = simulator_sensor.clone()
camera_sim_sensor.hfov = 90  # horizontal field of view in degrees
camera_sim_sensor.sensor_subtype = "PINHOLE"

simulator_depth_sensor = simulator_sensor.clone()
simulator_depth_sensor.min_depth = 0.0
simulator_depth_sensor.max_depth = 10.0
simulator_depth_sensor.normalize_depth = True

# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.rgb_sensor = camera_sim_sensor.clone()
_C.habitat.simulator.rgb_sensor.type = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.depth_sensor = camera_sim_sensor.clone()
_C.habitat.simulator.depth_sensor.merge_from_other_cfg(simulator_depth_sensor)
_C.habitat.simulator.depth_sensor.type = "HabitatSimDepthSensor"
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.semantic_sensor = camera_sim_sensor.clone()
_C.habitat.simulator.semantic_sensor.type = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# EQUIRECT RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.equirect_rgb_sensor = simulator_sensor.clone()
_C.habitat.simulator.equirect_rgb_sensor.type = (
    "HabitatSimEquirectangularRGBSensor"
)
# -----------------------------------------------------------------------------
# EQUIRECT DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.equirect_depth_sensor = simulator_sensor.clone()
_C.habitat.simulator.equirect_depth_sensor.merge_from_other_cfg(
    simulator_depth_sensor
)
_C.habitat.simulator.equirect_depth_sensor.type = (
    "HabitatSimEquirectangularDepthSensor"
)
# -----------------------------------------------------------------------------
# EQUIRECT SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.equirect_semantic_sensor = simulator_sensor.clone()
_C.habitat.simulator.equirect_semantic_sensor.type = (
    "HabitatSimEquirectangularSemanticSensor"
)
# -----------------------------------------------------------------------------
# FISHEYE SENSOR
# -----------------------------------------------------------------------------
fisheye_sim_sensor = simulator_sensor.clone()
fisheye_sim_sensor.height = fisheye_sim_sensor.width
# -----------------------------------------------------------------------------
# robot HEAD RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.head_rgb_sensor = _C.habitat.simulator.rgb_sensor.clone()
_C.habitat.simulator.head_rgb_sensor.uuid = "robot_head_rgb"
# -----------------------------------------------------------------------------
# robot HEAD DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.head_depth_sensor = (
    _C.habitat.simulator.depth_sensor.clone()
)
_C.habitat.simulator.head_depth_sensor.uuid = "robot_head_depth"
# -----------------------------------------------------------------------------
# ARM RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.arm_rgb_sensor = _C.habitat.simulator.rgb_sensor.clone()
_C.habitat.simulator.arm_rgb_sensor.uuid = "robot_arm_rgb"
# -----------------------------------------------------------------------------
# ARM DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.arm_depth_sensor = (
    _C.habitat.simulator.depth_sensor.clone()
)
_C.habitat.simulator.arm_depth_sensor.uuid = "robot_arm_depth"
# -----------------------------------------------------------------------------
# 3rd RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.third_rgb_sensor = _C.habitat.simulator.rgb_sensor.clone()
_C.habitat.simulator.third_rgb_sensor.uuid = "robot_third_rgb"
# -----------------------------------------------------------------------------
# 3rd DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.third_depth_sensor = (
    _C.habitat.simulator.depth_sensor.clone()
)
_C.habitat.simulator.third_depth_sensor.uuid = "robot_third_rgb"

# The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
# Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
# Camera Model, The International Conference on 3D Vision (3DV), 2018
# You can find the intrinsic parameters for the other lenses in the same table as well.
fisheye_sim_sensor.xi = -0.27
fisheye_sim_sensor.alpha = 0.57
fisheye_sim_sensor.focal_length = [364.84, 364.86]
# Place camera at center of screen
# Can be specified, otherwise is calculated automatically.
fisheye_sim_sensor.principal_point_offset = None  # (defaults to (h/2,w/2))
fisheye_sim_sensor.sensor_model_type = "DOUBLE_SPHERE"
# -----------------------------------------------------------------------------
# FISHEYE RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.fisheye_rgb_sensor = fisheye_sim_sensor.clone()
_C.habitat.simulator.fisheye_rgb_sensor.type = "HabitatSimFisheyeRGBSensor"
# -----------------------------------------------------------------------------
# FISHEYE DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.fisheye_depth_sensor = fisheye_sim_sensor.clone()
_C.habitat.simulator.fisheye_depth_sensor.merge_from_other_cfg(
    simulator_depth_sensor
)
_C.habitat.simulator.fisheye_depth_sensor.type = "HabitatSimFisheyeDepthSensor"
# -----------------------------------------------------------------------------
# FISHEYE SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.fisheye_semantic_sensor = fisheye_sim_sensor.clone()
_C.habitat.simulator.fisheye_semantic_sensor.type = (
    "HabitatSimFisheyeSemanticSensor"
)
# -----------------------------------------------------------------------------
# agent
# -----------------------------------------------------------------------------
_C.habitat.simulator.agent_0 = CN()
_C.habitat.simulator.agent_0.height = 1.5
_C.habitat.simulator.agent_0.radius = 0.1
_C.habitat.simulator.agent_0.sensors = ["rgb_sensor"]
_C.habitat.simulator.agent_0.is_set_start_state = False
_C.habitat.simulator.agent_0.start_position = [0, 0, 0]
_C.habitat.simulator.agent_0.start_rotation = [0, 0, 0, 1]
_C.habitat.simulator.agent_0.joint_start_noise = 0.0
_C.habitat.simulator.agent_0.robot_urdf = (
    "data/robots/hab_fetch/robots/hab_fetch.urdf"
)
_C.habitat.simulator.agent_0.robot_type = "FetchRobot"
_C.habitat.simulator.agent_0.ik_arm_urdf = (
    "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
)
_C.habitat.simulator.agents = ["agent_0"]
# -----------------------------------------------------------------------------
# SIMULATOR habitat_sim_v0
# -----------------------------------------------------------------------------
_C.habitat.simulator.habitat_sim_v0 = CN()
_C.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.habitat.simulator.habitat_sim_v0.gpu_gpu = False
# Whether or not the agent slides on collisions
_C.habitat.simulator.habitat_sim_v0.allow_sliding = True
_C.habitat.simulator.habitat_sim_v0.frustum_culling = True
_C.habitat.simulator.habitat_sim_v0.enable_physics = False
_C.habitat.simulator.habitat_sim_v0.physics_config_file = (
    "./data/default.physics_config.json"
)
# Possibly unstable optimization for extra performance with concurrent rendering
_C.habitat.simulator.habitat_sim_v0.leave_context_with_background_renderer = (
    False
)
_C.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = False
# -----------------------------------------------------------------------------
# pyrobot
# -----------------------------------------------------------------------------
_C.habitat.pyrobot = CN()
_C.habitat.pyrobot.robots = ["locobot"]  # types of robots supported
_C.habitat.pyrobot.robot = "locobot"
_C.habitat.pyrobot.sensors = ["rgb_sensor", "depth_sensor", "bump_sensor"]
_C.habitat.pyrobot.base_controller = "proportional"
_C.habitat.pyrobot.base_planner = "none"
# -----------------------------------------------------------------------------
# sensors
# -----------------------------------------------------------------------------
pyrobot_visual_sensor = CN()
pyrobot_visual_sensor.height = 480
pyrobot_visual_sensor.width = 640
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.rgb_sensor = pyrobot_visual_sensor.clone()
_C.habitat.pyrobot.rgb_sensor.type = "PyRobotRGBSensor"
_C.habitat.pyrobot.rgb_sensor.center_crop = False
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.depth_sensor = pyrobot_visual_sensor.clone()
_C.habitat.pyrobot.depth_sensor.type = "PyRobotDepthSensor"
_C.habitat.pyrobot.depth_sensor.min_depth = 0.0
_C.habitat.pyrobot.depth_sensor.max_depth = 5.0
_C.habitat.pyrobot.depth_sensor.normalize_depth = True
_C.habitat.pyrobot.depth_sensor.center_crop = False
# -----------------------------------------------------------------------------
# BUMP SENSOR
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.bump_sensor = CN()
_C.habitat.pyrobot.bump_sensor.type = "PyRobotBumpSensor"
# -----------------------------------------------------------------------------
# actions locobot
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.locobot = CN()
_C.habitat.pyrobot.locobot.actions = ["base_actions", "camera_actions"]
_C.habitat.pyrobot.locobot.base_actions = ["go_to_relative", "go_to_absolute"]
_C.habitat.pyrobot.locobot.camera_actions = [
    "set_pan",
    "set_tilt",
    "set_pan_tilt",
]
# TODO(akadian): add support for Arm actions
# -----------------------------------------------------------------------------
# dataset
# -----------------------------------------------------------------------------
_C.habitat.dataset = CN()
_C.habitat.dataset.type = "PointNav-v1"
_C.habitat.dataset.split = "train"
_C.habitat.dataset.scenes_dir = "data/scene_datasets"
_C.habitat.dataset.content_scenes = ["*"]
_C.habitat.dataset.data_path = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)

# -----------------------------------------------------------------------------
# gym
# -----------------------------------------------------------------------------
_C.habitat.gym = CN()
_C.habitat.gym.auto_name = ""
_C.habitat.gym.obs_keys = None
_C.habitat.gym.action_keys = None
_C.habitat.gym.achieved_goal_keys = []
_C.habitat.gym.desired_goal_keys = []

# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------
# GymHabitatEnv works for all Habitat tasks, including Navigation and Rearrange.
# To use a gym environment from the registry, use the GymRegistryEnv.
# Any other environment needs to be created and registered.
_C.habitat.env_task = "GymHabitatEnv"
# The dependencies for launching the GymRegistryEnv environments.
# modules listed here will be imported prior to making the environment with
# gym.make()
_C.habitat.env_task_gym_dependencies = []
# The key of the gym environment in the registry to use in GymRegistryEnv
# for example: `Cartpole-v0`
_C.habitat.env_task_gym_id = ""


def _get_full_config_path(config_path: str) -> str:
    if osp.exists(config_path):
        return config_path

    proposed_full_path = osp.join(_HABITAT_CFG_DIR, config_path)
    if osp.exists(proposed_full_path):
        return proposed_full_path

    raise RuntimeError(f"No file found for config '{config_path}'")


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config_path = _get_full_config_path(config_path)
            if not osp.exists(config_path):
                logger.warn(
                    f"Config file {config_path} could not be found. "
                    "Note that configuration files were moved to "
                    "the `habitat-lab/habitat/config` folder."
                )
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
