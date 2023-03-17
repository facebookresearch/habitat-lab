#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
python ./habitat_baselines/run.py \
--exp-config habitat_baselines/config/rearrange/gala_kinematic_local.yaml \
--run-type train \
SAVE_VIDEOS_INTERVAL 15 \
SIMULATOR.HEAD_RGB_SENSOR.WIDTH 256 \
SIMULATOR.HEAD_RGB_SENSOR.HEIGHT 256 \
SIMULATOR.HEAD_DEPTH_SENSOR.WIDTH 256 \
SIMULATOR.HEAD_DEPTH_SENSOR.HEIGHT 256 \
NUM_ENVIRONMENTS 4 \
NUM_UPDATES 60
