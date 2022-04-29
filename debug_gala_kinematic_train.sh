#!/usr/bin/env bash
python ./habitat_baselines/run.py \
--exp-config habitat_baselines/config/rearrange/gala_kinematic_local.yaml \
--run-type train \
SAVE_VIDEOS_INTERVAL 15 \
SIMULATOR.HEAD_RGB_SENSOR.WIDTH 512 \
SIMULATOR.HEAD_RGB_SENSOR.HEIGHT 512 \
SIMULATOR.HEAD_DEPTH_SENSOR.WIDTH 512 \
SIMULATOR.HEAD_DEPTH_SENSOR.HEIGHT 512 \
NUM_ENVIRONMENTS 4 \
NUM_UPDATES 60
