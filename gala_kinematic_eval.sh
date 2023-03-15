#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
python ./habitat_baselines/run.py \
--exp-config habitat_baselines/config/rearrange/gala_kinematic_local.yaml \
--run-type eval \
SENSORS "[\"DEBUG_RGB_SENSOR\", \"DEPTH_SENSOR\", \"RGB_SENSOR\", \"ROBOT_START_RELATIVE\", \"ROBOT_TARGET_RELATIVE\", \"EE_START_RELATIVE\", \"EE_TARGET_RELATIVE\", \"ROBOT_EE_RELATIVE\", \"ROBOT_GOAL_RELATIVE\", \"EE_GOAL_RELATIVE\", \"IS_HOLDING_SENSOR\", \"JOINT_SENSOR\", \"STEP_COUNT_SENSOR\"]"
