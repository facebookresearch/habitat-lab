#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import yacs.config


# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_C.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
_C.ENVIRONMENT.ITERATOR_OPTIONS = CN()
_C.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = int(1e4)
_C.ENVIRONMENT.ITERATOR_OPTIONS.STEP_REPETITION_RANGE = 0.2
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.REWARD_MEASURE = "distance_to_goal"
_C.TASK.SUCCESS_MEASURE = "spl"
_C.TASK.SUCCESS_REWARD = 2.5
_C.TASK.SLACK_REWARD = -0.01
_C.TASK.END_ON_SUCCESS = False
# -----------------------------------------------------------------------------
# # NAVIGATION TASK
# -----------------------------------------------------------------------------
_C.TASK.TYPE = "Nav-v0"
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = []
_C.TASK.GOAL_SENSOR_UUID = "pointgoal"
_C.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
# -----------------------------------------------------------------------------
# # REARRANGE TASK
# -----------------------------------------------------------------------------
_C.TASK.MAX_COLLISIONS = -1.0
_C.TASK.COUNT_OBJ_COLLISIONS = True
_C.TASK.COUNT_ROBOT_OBJ_COLLS = False
_C.TASK.SETTLE_STEPS = 5
_C.TASK.CONSTRAINT_VIOLATION_ENDS_EPISODE = True
_C.TASK.FORCE_REGENERATE = (
    False  # Forced to regenerate the starts even if they are already cached.
)
_C.TASK.SHOULD_SAVE_TO_CACHE = True  # Saves the generated starts to a cache if they are not already generated.
_C.TASK.MUST_LOOK_AT_TARG = True
_C.TASK.OBJECT_IN_HAND_SAMPLE_PROB = 0.167
_C.TASK.USING_SUBTASKS = False
_C.TASK.DEBUG_GOAL_POINT = True
_C.TASK.RENDER_TARGET = True
_C.TASK.EE_SAMPLE_FACTOR = 0.2
_C.TASK.EE_EXCLUDE_REGION = 0.0
# In radians
_C.TASK.BASE_ANGLE_NOISE = 0.15
_C.TASK.BASE_NOISE = 0.05
_C.TASK.SPAWN_REGION_SCALE = 0.2
_C.TASK.JOINT_MAX_IMPULSE = -1.0
_C.TASK.DESIRED_RESTING_POSITION = []
_C.TASK.USE_MARKER_T = True
_C.TASK.SUCCESS_STATE = 0.0
# Measurements for composite tasks.
_C.TASK.REWARD_MEASUREMENT = ""
_C.TASK.SUCCESS_MEASUREMENT = ""
# If true, does not care about navigability or collisions with objects when spawning
# robot
_C.TASK.EASY_INIT = False
_C.TASK.SHOULD_ENFORCE_TARGET_WITHIN_REACH = False
# -----------------------------------------------------------------------------
# # COMPOSITE TASK CONFIG
# -----------------------------------------------------------------------------
_C.TASK.TASK_SPEC_BASE_PATH = "configs/tasks/rearrange/pddl/"
_C.TASK.TASK_SPEC = "nav_pick"
# PDDL domain params
_C.TASK.PDDL_DOMAIN_DEF = (
    "configs/tasks/rearrange/pddl/replica_cad_domain.yaml"
)
_C.TASK.OBJ_SUCC_THRESH = 0.3
_C.TASK.ART_SUCC_THRESH = 0.15
_C.TASK.SINGLE_EVAL_NODE = -1
_C.TASK.LIMIT_TASK_NODE = -1  # delete
_C.TASK.LIMIT_TASK_LEN_SCALING = 0.0  # delete
_C.TASK.DEBUG_SKIP_TO_NODE = -1
_C.TASK.SKIP_NODES = ["move_obj"]
_C.TASK.FILTER_NAV_TO_TASKS = []
# -----------------------------------------------------------------------------
# # ACTIONS
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS = CN()
_C.TASK.ACTIONS.STOP = CN()
_C.TASK.ACTIONS.STOP.TYPE = "StopAction"
_C.TASK.ACTIONS.EMPTY = CN()
_C.TASK.ACTIONS.EMPTY.TYPE = "EmptyAction"
# -----------------------------------------------------------------------------
# # NAVIGATION ACTIONS
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.MOVE_FORWARD = CN()
_C.TASK.ACTIONS.MOVE_FORWARD.TYPE = "MoveForwardAction"
_C.TASK.ACTIONS.TURN_LEFT = CN()
_C.TASK.ACTIONS.TURN_LEFT.TYPE = "TurnLeftAction"
_C.TASK.ACTIONS.TURN_RIGHT = CN()
_C.TASK.ACTIONS.TURN_RIGHT.TYPE = "TurnRightAction"
_C.TASK.ACTIONS.LOOK_UP = CN()
_C.TASK.ACTIONS.LOOK_UP.TYPE = "LookUpAction"
_C.TASK.ACTIONS.LOOK_DOWN = CN()
_C.TASK.ACTIONS.LOOK_DOWN.TYPE = "LookDownAction"
_C.TASK.ACTIONS.TELEPORT = CN()
_C.TASK.ACTIONS.TELEPORT.TYPE = "TeleportAction"
_C.TASK.ACTIONS.VELOCITY_CONTROL = CN()
_C.TASK.ACTIONS.VELOCITY_CONTROL.TYPE = "VelocityAction"
_C.TASK.ACTIONS.VELOCITY_CONTROL.LIN_VEL_RANGE = [0.0, 0.25]  # meters per sec
_C.TASK.ACTIONS.VELOCITY_CONTROL.ANG_VEL_RANGE = [-10.0, 10.0]  # deg per sec
_C.TASK.ACTIONS.VELOCITY_CONTROL.MIN_ABS_LIN_SPEED = 0.025  # meters per sec
_C.TASK.ACTIONS.VELOCITY_CONTROL.MIN_ABS_ANG_SPEED = 1.0  # deg per sec
_C.TASK.ACTIONS.VELOCITY_CONTROL.TIME_STEP = 1.0  # seconds
# -----------------------------------------------------------------------------
# # REARRANGE ACTIONS
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.ARM_ACTION = CN()
_C.TASK.ACTIONS.ARM_ACTION.TYPE = "ArmAction"
_C.TASK.ACTIONS.ARM_ACTION.ARM_CONTROLLER = "ArmRelPosAction"
_C.TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER = None
_C.TASK.ACTIONS.ARM_ACTION.ARM_JOINT_DIMENSIONALITY = 7
_C.TASK.ACTIONS.ARM_ACTION.GRASP_THRESH_DIST = 0.15
_C.TASK.ACTIONS.ARM_ACTION.DISABLE_GRIP = False
_C.TASK.ACTIONS.ARM_ACTION.DELTA_POS_LIMIT = 0.0125
_C.TASK.ACTIONS.ARM_ACTION.EE_CTRL_LIM = 0.015
_C.TASK.ACTIONS.ARM_ACTION.SHOULD_CLIP = False
_C.TASK.ACTIONS.ARM_ACTION.RENDER_EE_TARGET = False
_C.TASK.ACTIONS.ARM_ACTION.ORACLE_GRASP = False
_C.TASK.ACTIONS.BASE_VELOCITY = CN()
_C.TASK.ACTIONS.BASE_VELOCITY.TYPE = "BaseVelAction"
_C.TASK.ACTIONS.BASE_VELOCITY.LIN_SPEED = 12.0
_C.TASK.ACTIONS.BASE_VELOCITY.ANG_SPEED = 12.0
_C.TASK.ACTIONS.BASE_VELOCITY.ALLOW_DYN_SLIDE = True
_C.TASK.ACTIONS.BASE_VELOCITY.END_ON_STOP = False
_C.TASK.ACTIONS.BASE_VELOCITY.ALLOW_BACK = True
_C.TASK.ACTIONS.BASE_VELOCITY.MIN_ABS_LIN_SPEED = 1.0
_C.TASK.ACTIONS.BASE_VELOCITY.MIN_ABS_ANG_SPEED = 1.0
_C.TASK.ACTIONS.REARRANGE_STOP = CN()
_C.TASK.ACTIONS.REARRANGE_STOP.TYPE = "RearrangeStopAction"
# -----------------------------------------------------------------------------
# # TASK SENSORS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_SENSOR = CN()
_C.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
_C.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR = _C.TASK.POINTGOAL_SENSOR.clone()
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.TYPE = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# OBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECTGOAL_SENSOR = CN()
_C.TASK.OBJECTGOAL_SENSOR.TYPE = "ObjectGoalSensor"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC = "TASK_CATEGORY_ID"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC_MAX_VAL = 50
# -----------------------------------------------------------------------------
# IMAGEGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.IMAGEGOAL_SENSOR = CN()
_C.TASK.IMAGEGOAL_SENSOR.TYPE = "ImageGoalSensor"
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"
# -----------------------------------------------------------------------------
# COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COMPASS_SENSOR = CN()
_C.TASK.COMPASS_SENSOR.TYPE = "CompassSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_SENSOR = CN()
_C.TASK.GPS_SENSOR.TYPE = "GPSSensor"
_C.TASK.GPS_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROXIMITY_SENSOR = CN()
_C.TASK.PROXIMITY_SENSOR.TYPE = "ProximitySensor"
_C.TASK.PROXIMITY_SENSOR.MAX_DETECTION_RADIUS = 2.0
# -----------------------------------------------------------------------------
# JOINT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.JOINT_SENSOR = CN()
_C.TASK.JOINT_SENSOR.TYPE = "JointSensor"
_C.TASK.JOINT_SENSOR.DIMENSIONALITY = 7
# -----------------------------------------------------------------------------
# END EFFECTOR POSITION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.END_EFFECTOR_SENSOR = CN()
_C.TASK.END_EFFECTOR_SENSOR.TYPE = "EEPositionSensor"
# -----------------------------------------------------------------------------
# IS HOLDING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.IS_HOLDING_SENSOR = CN()
_C.TASK.IS_HOLDING_SENSOR.TYPE = "IsHoldingSensor"
# -----------------------------------------------------------------------------
# RELATIVE RESTING POSISITON SENSOR
# -----------------------------------------------------------------------------
_C.TASK.RELATIVE_RESTING_POS_SENSOR = CN()
_C.TASK.RELATIVE_RESTING_POS_SENSOR.TYPE = "RelativeRestingPositionSensor"
# -----------------------------------------------------------------------------
# JOINT VELOCITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.JOINT_VELOCITY_SENSOR = CN()
_C.TASK.JOINT_VELOCITY_SENSOR.TYPE = "JointVelocitySensor"
_C.TASK.JOINT_VELOCITY_SENSOR.DIMENSIONALITY = 7
# -----------------------------------------------------------------------------
# ORACLE NAVIGATION ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_NAV_ACTION_SENSOR = CN()
_C.TASK.ORACLE_NAV_ACTION_SENSOR.TYPE = "OracleNavigationActionSensor"
# -----------------------------------------------------------------------------
# RESTING POSITION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.RESTING_POS_SENSOR = CN()
_C.TASK.RESTING_POS_SENSOR.TYPE = "RestingPositionSensor"
# -----------------------------------------------------------------------------
# ART JOINT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ART_JOINT_SENSOR = CN()
_C.TASK.ART_JOINT_SENSOR.TYPE = "ArtJointSensor"
# -----------------------------------------------------------------------------
# NAV GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.NAV_GOAL_SENSOR = CN()
_C.TASK.NAV_GOAL_SENSOR.TYPE = "NavGoalSensor"
# -----------------------------------------------------------------------------
# ART JOINT NO VELOCITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ART_JOINT_SENSOR_NO_VEL = CN()
_C.TASK.ART_JOINT_SENSOR_NO_VEL.TYPE = "ArtJointSensorNoVel"
# -----------------------------------------------------------------------------
# MARKER RELATIVE POSISITON SENSOR
# -----------------------------------------------------------------------------
_C.TASK.MARKER_REL_POS_SENSOR = CN()
_C.TASK.MARKER_REL_POS_SENSOR.TYPE = "MarkerRelPosSensor"
# -----------------------------------------------------------------------------
# TARGET START SENSOR
# -----------------------------------------------------------------------------
_C.TASK.TARGET_START_SENSOR = CN()
_C.TASK.TARGET_START_SENSOR.TYPE = "TargetStartSensor"
_C.TASK.TARGET_START_SENSOR.GOAL_FORMAT = "CARTESIAN"
_C.TASK.TARGET_START_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# OBJECT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_SENSOR = CN()
_C.TASK.OBJECT_SENSOR.TYPE = "TargetCurrentSensor"
_C.TASK.OBJECT_SENSOR.GOAL_FORMAT = "CARTESIAN"
_C.TASK.OBJECT_SENSOR.DIMENSIONALITY = 3

# -----------------------------------------------------------------------------
# GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GOAL_SENSOR = CN()
_C.TASK.GOAL_SENSOR.TYPE = "GoalSensor"
_C.TASK.GOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
_C.TASK.GOAL_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# TARGET START OR GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.TARGET_START_POINT_GOAL_SENSOR = CN()
_C.TASK.TARGET_START_POINT_GOAL_SENSOR.TYPE = (
    "TargetOrGoalStartPointGoalSensor"
)
# -----------------------------------------------------------------------------
# TARGET START GPS/COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.TARGET_START_GPS_COMPASS_SENSOR = CN()
_C.TASK.TARGET_START_GPS_COMPASS_SENSOR.TYPE = "TargetStartGpsCompassSensor"
# -----------------------------------------------------------------------------
# TARGET GOAL GPS/COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.TARGET_GOAL_GPS_COMPASS_SENSOR = CN()
_C.TASK.TARGET_GOAL_GPS_COMPASS_SENSOR.TYPE = "TargetGoalGpsCompassSensor"
# -----------------------------------------------------------------------------
# NAV TO SKILL ID SENSOR
# -----------------------------------------------------------------------------
_C.TASK.NAV_TO_SKILL_SENSOR = CN()
_C.TASK.NAV_TO_SKILL_SENSOR.TYPE = "NavToSkillSensor"
_C.TASK.NAV_TO_SKILL_SENSOR.NUM_SKILLS = 8
# -----------------------------------------------------------------------------
# ABSOLUTE TARGET START SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ABS_TARGET_START_SENSOR = CN()
_C.TASK.ABS_TARGET_START_SENSOR.TYPE = "AbsTargetStartSensor"
_C.TASK.ABS_TARGET_START_SENSOR.GOAL_FORMAT = "CARTESIAN"
_C.TASK.ABS_TARGET_START_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# ABSOLUTE GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ABS_GOAL_SENSOR = CN()
_C.TASK.ABS_GOAL_SENSOR.TYPE = "AbsGoalSensor"
_C.TASK.ABS_GOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
_C.TASK.ABS_GOAL_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# DISTANCE TO NAVIGATION GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.DIST_TO_NAV_GOAL = CN()
_C.TASK.DIST_TO_NAV_GOAL.TYPE = "DistToNavGoalSensor"
# -----------------------------------------------------------------------------
# LOCALIZATION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.LOCALIZATION_SENSOR = CN()
_C.TASK.LOCALIZATION_SENSOR.TYPE = "LocalizationSensor"
# -----------------------------------------------------------------------------
# SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SUCCESS = CN()
_C.TASK.SUCCESS.TYPE = "Success"
_C.TASK.SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPL = CN()
_C.TASK.SPL.TYPE = "SPL"
# -----------------------------------------------------------------------------
# SOFT-SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SOFT_SPL = CN()
_C.TASK.SOFT_SPL.TYPE = "SoftSPL"
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1024
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
_C.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True
# -----------------------------------------------------------------------------
# COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISIONS = CN()
_C.TASK.COLLISIONS.TYPE = "Collisions"
# -----------------------------------------------------------------------------
# GENERAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ROBOT_FORCE = CN()
_C.TASK.ROBOT_FORCE.TYPE = "RobotForce"
_C.TASK.ROBOT_FORCE.MIN_FORCE = 20.0

_C.TASK.FORCE_TERMINATE = CN()
_C.TASK.FORCE_TERMINATE.TYPE = "ForceTerminate"
_C.TASK.FORCE_TERMINATE.MAX_ACCUM_FORCE = -1.0

_C.TASK.ROBOT_COLLS = CN()
_C.TASK.ROBOT_COLLS.TYPE = "RobotCollisions"
_C.TASK.OBJECT_TO_GOAL_DISTANCE = CN()
_C.TASK.OBJECT_TO_GOAL_DISTANCE.TYPE = "ObjectToGoalDistance"
_C.TASK.END_EFFECTOR_TO_OBJECT_DISTANCE = CN()
_C.TASK.END_EFFECTOR_TO_OBJECT_DISTANCE.TYPE = "EndEffectorToObjectDistance"
_C.TASK.END_EFFECTOR_TO_REST_DISTANCE = CN()
_C.TASK.END_EFFECTOR_TO_REST_DISTANCE.TYPE = "EndEffectorToRestDistance"

_C.TASK.ART_OBJ_AT_DESIRED_STATE = CN()
_C.TASK.ART_OBJ_AT_DESIRED_STATE.TYPE = "ArtObjAtDesiredState"
_C.TASK.ART_OBJ_AT_DESIRED_STATE.USE_ABSOLUTE_DISTANCE = True
_C.TASK.ART_OBJ_AT_DESIRED_STATE.SUCCESS_DIST_THRESHOLD = 0.05

_C.TASK.EE_DIST_TO_MARKER = CN()
_C.TASK.EE_DIST_TO_MARKER.TYPE = "EndEffectorDistToMarker"
_C.TASK.ART_OBJ_STATE = CN()
_C.TASK.ART_OBJ_STATE.TYPE = "ArtObjState"
_C.TASK.ART_OBJ_SUCCESS = CN()
_C.TASK.ART_OBJ_SUCCESS.TYPE = "ArtObjSuccess"
_C.TASK.ART_OBJ_SUCCESS.REST_DIST_THRESHOLD = 0.15

_C.TASK.ART_OBJ_REWARD = CN()
_C.TASK.ART_OBJ_REWARD.TYPE = "ArtObjReward"
_C.TASK.ART_OBJ_REWARD.DIST_REWARD = 1.0
_C.TASK.ART_OBJ_REWARD.WRONG_GRASP_END = False
_C.TASK.ART_OBJ_REWARD.WRONG_GRASP_PEN = 5.0
_C.TASK.ART_OBJ_REWARD.ART_DIST_REWARD = 10.0
_C.TASK.ART_OBJ_REWARD.EE_DIST_REWARD = 10.0
_C.TASK.ART_OBJ_REWARD.MARKER_DIST_REWARD = 0.0
_C.TASK.ART_OBJ_REWARD.ART_AT_DESIRED_STATE_REWARD = 5.0
_C.TASK.ART_OBJ_REWARD.GRASP_REWARD = 0.0
# General Rearrange Reward config
_C.TASK.ART_OBJ_REWARD.CONSTRAINT_VIOLATE_PEN = 10.0
_C.TASK.ART_OBJ_REWARD.FORCE_PEN = 0.0
_C.TASK.ART_OBJ_REWARD.MAX_FORCE_PEN = 1.0
_C.TASK.ART_OBJ_REWARD.FORCE_END_PEN = 10.0
# -----------------------------------------------------------------------------
# NAVIGATION MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ROT_DIST_TO_GOAL = CN()
_C.TASK.ROT_DIST_TO_GOAL.TYPE = "RotDistToGoal"
_C.TASK.DIST_TO_GOAL = CN()
_C.TASK.DIST_TO_GOAL.TYPE = "DistToGoal"
_C.TASK.BAD_CALLED_TERMINATE = CN()
_C.TASK.BAD_CALLED_TERMINATE.TYPE = "BadCalledTerminate"
_C.TASK.BAD_CALLED_TERMINATE.BAD_TERM_PEN = 0.0
_C.TASK.BAD_CALLED_TERMINATE.DECAY_BAD_TERM = False
_C.TASK.NAV_TO_POS_SUCC = CN()
_C.TASK.NAV_TO_POS_SUCC.TYPE = "NavToPosSucc"
_C.TASK.NAV_TO_POS_SUCC.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# REARRANGE NAVIGATION MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD = CN()
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.TYPE = "NavToObjReward"
# Reward the agent for facing the object?
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.SHOULD_REWARD_TURN = True
# What distance do we start giving the reward for facing the object?
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.TURN_REWARD_DIST = 0.1
# Multiplier on the angle distance to the goal.
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.ANGLE_DIST_REWARD = 1.0
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.DIST_REWARD = 10.0
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.CONSTRAINT_VIOLATE_PEN = 10.0
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.FORCE_PEN = 0.0
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.MAX_FORCE_PEN = 1.0
_C.TASK.REARRANGE_NAV_TO_OBJ_REWARD.FORCE_END_PEN = 10.0

_C.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS = CN()
_C.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS.TYPE = "NavToObjSuccess"
_C.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS.MUST_LOOK_AT_TARG = True
_C.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS.MUST_CALL_STOP = True
# Distance in radians.
_C.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS.SUCCESS_ANGLE_DIST = 0.15
_C.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS.HEURISTIC_STOP = False
# -----------------------------------------------------------------------------
# REARRANGE REACH MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.REARRANGE_REACH_REWARD = CN()
_C.TASK.REARRANGE_REACH_REWARD.TYPE = "RearrangeReachReward"
_C.TASK.REARRANGE_REACH_REWARD.SCALE = 1.0
_C.TASK.REARRANGE_REACH_REWARD.DIFF_REWARD = True
_C.TASK.REARRANGE_REACH_REWARD.SPARSE_REWARD = False

_C.TASK.REARRANGE_REACH_SUCCESS = CN()
_C.TASK.REARRANGE_REACH_SUCCESS.TYPE = "RearrangeReachSuccess"
_C.TASK.REARRANGE_REACH_SUCCESS.SUCC_THRESH = 0.2
# -----------------------------------------------------------------------------
# NUM STEPS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.NUM_STEPS = CN()
_C.TASK.NUM_STEPS.TYPE = "NumStepsMeasure"
# -----------------------------------------------------------------------------
# DID PICK OBJECT MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DID_PICK_OBJECT = CN()
_C.TASK.DID_PICK_OBJECT.TYPE = "DidPickObjectMeasure"
# -----------------------------------------------------------------------------
# DID VIOLATE HOLD CONSTRAINT MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DID_VIOLATE_HOLD_CONSTRAINT = CN()
_C.TASK.DID_VIOLATE_HOLD_CONSTRAINT.TYPE = "DidViolateHoldConstraintMeasure"
# -----------------------------------------------------------------------------
# MOVE OBJECTS REWARD
# -----------------------------------------------------------------------------
_C.TASK.MOVE_OBJECTS_REWARD = CN()
_C.TASK.MOVE_OBJECTS_REWARD.TYPE = "MoveObjectsReward"
_C.TASK.MOVE_OBJECTS_REWARD.PICK_REWARD = 1.0
_C.TASK.MOVE_OBJECTS_REWARD.SUCCESS_DIST = 0.15
_C.TASK.MOVE_OBJECTS_REWARD.SINGLE_REARRANGE_REWARD = 1.0
_C.TASK.MOVE_OBJECTS_REWARD.DIST_REWARD = 1.0
_C.TASK.MOVE_OBJECTS_REWARD.CONSTRAINT_VIOLATE_PEN = 10.0
_C.TASK.MOVE_OBJECTS_REWARD.FORCE_PEN = 0.001
_C.TASK.MOVE_OBJECTS_REWARD.MAX_FORCE_PEN = 1.0
_C.TASK.MOVE_OBJECTS_REWARD.FORCE_END_PEN = 10.0
# -----------------------------------------------------------------------------
# PICK MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PICK_REWARD = CN()
_C.TASK.PICK_REWARD.TYPE = "RearrangePickReward"
_C.TASK.PICK_REWARD.DIST_REWARD = 20.0
_C.TASK.PICK_REWARD.SUCC_REWARD = 10.0
_C.TASK.PICK_REWARD.PICK_REWARD = 20.0
_C.TASK.PICK_REWARD.CONSTRAINT_VIOLATE_PEN = 10.0
_C.TASK.PICK_REWARD.DROP_PEN = 5.0
_C.TASK.PICK_REWARD.WRONG_PICK_PEN = 5.0
_C.TASK.PICK_REWARD.COLL_PEN = 1.0
_C.TASK.PICK_REWARD.ROBOT_OBJ_COLL_PEN = 0.0
_C.TASK.PICK_REWARD.MAX_ACCUM_FORCE = 5000.0
_C.TASK.PICK_REWARD.FORCE_PEN = 0.001
_C.TASK.PICK_REWARD.MAX_FORCE_PEN = 1.0
_C.TASK.PICK_REWARD.FORCE_END_PEN = 10.0
_C.TASK.PICK_REWARD.USE_DIFF = True
_C.TASK.PICK_REWARD.DROP_OBJ_SHOULD_END = False
_C.TASK.PICK_REWARD.WRONG_PICK_SHOULD_END = False
_C.TASK.PICK_REWARD.COLLISION_PENALTY = 0.0
_C.TASK.PICK_REWARD.ROBOT_OBJ_COLLISION_PENALTY = 0.0
_C.TASK.PICK_SUCCESS = CN()
_C.TASK.PICK_SUCCESS.TYPE = "RearrangePickSuccess"
_C.TASK.PICK_SUCCESS.EE_RESTING_SUCCESS_THRESHOLD = 0.15
# -----------------------------------------------------------------------------
# PLACE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.OBJ_AT_GOAL = CN()
_C.TASK.OBJ_AT_GOAL.TYPE = "ObjAtGoal"
_C.TASK.OBJ_AT_GOAL.SUCC_THRESH = 0.15

_C.TASK.PLACE_REWARD = CN()
_C.TASK.PLACE_REWARD.TYPE = "PlaceReward"
_C.TASK.PLACE_REWARD.DIST_REWARD = 20.0
_C.TASK.PLACE_REWARD.SUCC_REWARD = 10.0
_C.TASK.PLACE_REWARD.PLACE_REWARD = 20.0
_C.TASK.PLACE_REWARD.DROP_PEN = 5.0
_C.TASK.PLACE_REWARD.USE_DIFF = True
_C.TASK.PLACE_REWARD.WRONG_DROP_SHOULD_END = False
_C.TASK.PLACE_REWARD.CONSTRAINT_VIOLATE_PEN = 10.0
_C.TASK.PLACE_REWARD.FORCE_PEN = 0.001
_C.TASK.PLACE_REWARD.MAX_FORCE_PEN = 1.0
_C.TASK.PLACE_REWARD.FORCE_END_PEN = 10.0

_C.TASK.PLACE_SUCCESS = CN()
_C.TASK.PLACE_SUCCESS.TYPE = "PlaceSuccess"
_C.TASK.PLACE_SUCCESS.EE_RESTING_SUCCESS_THRESHOLD = 0.15
# -----------------------------------------------------------------------------
# COMPOSITE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COMPOSITE_NODE_IDX = CN()
_C.TASK.COMPOSITE_NODE_IDX.TYPE = "CompositeNodeIdx"
_C.TASK.COMPOSITE_SUCCESS = CN()
_C.TASK.COMPOSITE_SUCCESS.TYPE = "CompositeSuccess"
_C.TASK.COMPOSITE_REWARD = CN()
_C.TASK.COMPOSITE_REWARD.TYPE = "CompositeReward"
_C.TASK.COMPOSITE_REWARD.STAGE_COMPLETE_REWARD = 10.0
_C.TASK.COMPOSITE_REWARD.SUCCESS_REWARD = 10.0
_C.TASK.DOES_WANT_TERMINATE = CN()
_C.TASK.DOES_WANT_TERMINATE.TYPE = "DoesWantTerminate"
_C.TASK.COMPOSITE_BAD_CALLED_TERMINATE = CN()
_C.TASK.COMPOSITE_BAD_CALLED_TERMINATE.TYPE = "CompositeBadCalledTerminate"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # EQA TASK
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.ANSWER = CN()
_C.TASK.ACTIONS.ANSWER.TYPE = "AnswerAction"
# # EQA TASK QUESTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.QUESTION_SENSOR = CN()
_C.TASK.QUESTION_SENSOR.TYPE = "QuestionSensor"
# -----------------------------------------------------------------------------
# # EQA TASK CORRECT_ANSWER measure for training
# -----------------------------------------------------------------------------
_C.TASK.CORRECT_ANSWER = CN()
_C.TASK.CORRECT_ANSWER.TYPE = "CorrectAnswer"
# -----------------------------------------------------------------------------
# # EQA TASK ANSWER SENSOR
# -----------------------------------------------------------------------------
_C.TASK.EPISODE_INFO = CN()
_C.TASK.EPISODE_INFO.TYPE = "EpisodeInfo"
# -----------------------------------------------------------------------------
# # VLN TASK INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.INSTRUCTION_SENSOR = CN()
_C.TASK.INSTRUCTION_SENSOR.TYPE = "InstructionSensor"
_C.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_GOAL = CN()
_C.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_C.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # ANSWER_ACCURACY MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ANSWER_ACCURACY = CN()
_C.TASK.ANSWER_ACCURACY.TYPE = "AnswerAccuracy"
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.CREATE_RENDERER = False
_C.SIMULATOR.REQUIRES_TEXTURES = True
_C.SIMULATOR.LAG_OBSERVATIONS = 0
_C.SIMULATOR.AUTO_SLEEP = False
_C.SIMULATOR.STEP_PHYSICS = True
_C.SIMULATOR.UPDATE_ROBOT = True
_C.SIMULATOR.CONCUR_RENDER = False
_C.SIMULATOR.NEEDS_MARKERS = (
    True  # If markers should be updated at every step.
)
_C.SIMULATOR.UPDATE_ROBOT = (
    True  # If the robot camera positions should be updated at every step.
)
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_C.SIMULATOR.SCENE_DATASET = "default"  # the scene dataset to load in the MetaDataMediator. Should contain SIMULATOR.SCENE
_C.SIMULATOR.ADDITIONAL_OBJECT_PATHS = (
    []
)  # a list of directory or config paths to search in addition to the dataset for object configs. Should match the generated episodes for the task.
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # angle to rotate left or right in degrees
_C.SIMULATOR.TILT_ANGLE = 15  # angle to tilt the camera up or down in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
_C.SIMULATOR.DEBUG_RENDER = False
# If in render mode a visualization of the rearrangement goal position should
# also be displayed.
_C.SIMULATOR.DEBUG_RENDER_GOAL = True
_C.SIMULATOR.ROBOT_JOINT_START_NOISE = 0.0
# Rearrange Agent Setup
_C.SIMULATOR.ARM_REST = [0.6, 0.0, 0.9]
_C.SIMULATOR.CTRL_FREQ = 120.0
_C.SIMULATOR.AC_FREQ_RATIO = 4
_C.SIMULATOR.ROBOT_URDF = "data/robots/hab_fetch/robots/hab_fetch.urdf"
_C.SIMULATOR.ROBOT_TYPE = "FetchRobot"
_C.SIMULATOR.EE_LINK_NAME = None
_C.SIMULATOR.LOAD_OBJS = False
# Rearrange Agent Grasping
_C.SIMULATOR.HOLD_THRESH = 0.09
_C.SIMULATOR.GRASP_IMPULSE = 1000.0
# ROBOT
_C.SIMULATOR.IK_ARM_URDF = "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
# -----------------------------------------------------------------------------
# SIMULATOR SENSORS
# -----------------------------------------------------------------------------
SIMULATOR_SENSOR = CN()
SIMULATOR_SENSOR.HEIGHT = 480
SIMULATOR_SENSOR.WIDTH = 640
SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]
SIMULATOR_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler's angles

# -----------------------------------------------------------------------------
# CAMERA SENSOR
# -----------------------------------------------------------------------------
CAMERA_SIM_SENSOR = SIMULATOR_SENSOR.clone()
CAMERA_SIM_SENSOR.HFOV = 90  # horizontal field of view in degrees
CAMERA_SIM_SENSOR.SENSOR_SUBTYPE = "PINHOLE"

SIMULATOR_DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
SIMULATOR_DEPTH_SENSOR.MIN_DEPTH = 0.0
SIMULATOR_DEPTH_SENSOR.MAX_DEPTH = 10.0
SIMULATOR_DEPTH_SENSOR.NORMALIZE_DEPTH = True

# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# EQUIRECT RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_RGB_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_RGB_SENSOR.TYPE = "HabitatSimEquirectangularRGBSensor"
# -----------------------------------------------------------------------------
# EQUIRECT DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR.TYPE = (
    "HabitatSimEquirectangularDepthSensor"
)
# -----------------------------------------------------------------------------
# EQUIRECT SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_SEMANTIC_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_SEMANTIC_SENSOR.TYPE = (
    "HabitatSimEquirectangularSemanticSensor"
)
# -----------------------------------------------------------------------------
# FISHEYE SENSOR
# -----------------------------------------------------------------------------
FISHEYE_SIM_SENSOR = SIMULATOR_SENSOR.clone()
FISHEYE_SIM_SENSOR.HEIGHT = FISHEYE_SIM_SENSOR.WIDTH
# -----------------------------------------------------------------------------
# ROBOT HEAD RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.HEAD_RGB_SENSOR.UUID = "robot_head_rgb"
# -----------------------------------------------------------------------------
# ROBOT HEAD DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.HEAD_DEPTH_SENSOR.UUID = "robot_head_depth"
# -----------------------------------------------------------------------------
# ARM RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.ARM_RGB_SENSOR.UUID = "robot_arm_rgb"
# -----------------------------------------------------------------------------
# ARM DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.ARM_DEPTH_SENSOR.UUID = "robot_arm_depth"
# -----------------------------------------------------------------------------
# 3rd RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.THIRD_RGB_SENSOR.UUID = "robot_third_rgb"
# -----------------------------------------------------------------------------
# 3rd DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.THIRD_DEPTH_SENSOR.UUID = "robot_third_rgb"

# The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
# Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
# Camera Model, The International Conference on 3D Vision (3DV), 2018
# You can find the intrinsic parameters for the other lenses in the same table as well.
FISHEYE_SIM_SENSOR.XI = -0.27
FISHEYE_SIM_SENSOR.ALPHA = 0.57
FISHEYE_SIM_SENSOR.FOCAL_LENGTH = [364.84, 364.86]
# Place camera at center of screen
# Can be specified, otherwise is calculated automatically.
FISHEYE_SIM_SENSOR.PRINCIPAL_POINT_OFFSET = None  # (defaults to (h/2,w/2))
FISHEYE_SIM_SENSOR.SENSOR_MODEL_TYPE = "DOUBLE_SPHERE"
# -----------------------------------------------------------------------------
# FISHEYE RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_RGB_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_RGB_SENSOR.TYPE = "HabitatSimFisheyeRGBSensor"
# -----------------------------------------------------------------------------
# FISHEYE DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR.TYPE = "HabitatSimFisheyeDepthSensor"
# -----------------------------------------------------------------------------
# FISHEYE SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_SEMANTIC_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_SEMANTIC_SENSOR.TYPE = "HabitatSimFisheyeSemanticSensor"
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
_C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
_C.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
_C.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
_C.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
_C.SIMULATOR.AGENTS = ["AGENT_0"]
# -----------------------------------------------------------------------------
# SIMULATOR HABITAT_SIM_V0
# -----------------------------------------------------------------------------
_C.SIMULATOR.HABITAT_SIM_V0 = CN()
_C.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
# Whether or not the agent slides on collisions
_C.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
_C.SIMULATOR.HABITAT_SIM_V0.FRUSTUM_CULLING = True
_C.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = False
_C.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = (
    "./data/default.physics_config.json"
)
# Possibly unstable optimization for extra performance with concurrent rendering
_C.SIMULATOR.HABITAT_SIM_V0.LEAVE_CONTEXT_WITH_BACKGROUND_RENDERER = False
# -----------------------------------------------------------------------------
# PYROBOT
# -----------------------------------------------------------------------------
_C.PYROBOT = CN()
_C.PYROBOT.ROBOTS = ["locobot"]  # types of robots supported
_C.PYROBOT.ROBOT = "locobot"
_C.PYROBOT.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "BUMP_SENSOR"]
_C.PYROBOT.BASE_CONTROLLER = "proportional"
_C.PYROBOT.BASE_PLANNER = "none"
# -----------------------------------------------------------------------------
# SENSORS
# -----------------------------------------------------------------------------
PYROBOT_VISUAL_SENSOR = CN()
PYROBOT_VISUAL_SENSOR.HEIGHT = 480
PYROBOT_VISUAL_SENSOR.WIDTH = 640
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.RGB_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.RGB_SENSOR.TYPE = "PyRobotRGBSensor"
_C.PYROBOT.RGB_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.DEPTH_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.DEPTH_SENSOR.TYPE = "PyRobotDepthSensor"
_C.PYROBOT.DEPTH_SENSOR.MIN_DEPTH = 0.0
_C.PYROBOT.DEPTH_SENSOR.MAX_DEPTH = 5.0
_C.PYROBOT.DEPTH_SENSOR.NORMALIZE_DEPTH = True
_C.PYROBOT.DEPTH_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# BUMP SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.BUMP_SENSOR = CN()
_C.PYROBOT.BUMP_SENSOR.TYPE = "PyRobotBumpSensor"
# -----------------------------------------------------------------------------
# ACTIONS LOCOBOT
# -----------------------------------------------------------------------------
_C.PYROBOT.LOCOBOT = CN()
_C.PYROBOT.LOCOBOT.ACTIONS = ["BASE_ACTIONS", "CAMERA_ACTIONS"]
_C.PYROBOT.LOCOBOT.BASE_ACTIONS = ["go_to_relative", "go_to_absolute"]
_C.PYROBOT.LOCOBOT.CAMERA_ACTIONS = ["set_pan", "set_tilt", "set_pan_tilt"]
# TODO(akadian): add support for Arm actions
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = "PointNav-v1"
_C.DATASET.SPLIT = "train"
_C.DATASET.SCENES_DIR = "data/scene_datasets"
_C.DATASET.CONTENT_SCENES = ["*"]
_C.DATASET.DATA_PATH = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)

# -----------------------------------------------------------------------------
# GYM
# -----------------------------------------------------------------------------
_C.GYM = CN()
_C.GYM.AUTO_NAME = ""
_C.GYM.CLASS_NAME = "RearrangeRLEnv"
_C.GYM.OBS_KEYS = None
_C.GYM.ACTION_KEYS = None
_C.GYM.ACHIEVED_GOAL_KEYS = []
_C.GYM.DESIRED_GOAL_KEYS = []
_C.GYM.FIX_INFO_DICT = True

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# DEPRECATED KEYS
# -----------------------------------------------------------------------------'
_C.register_deprecated_key("TASK.SUCCESS_DISTANCE")
# -----------------------------------------------------------------------------


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
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
