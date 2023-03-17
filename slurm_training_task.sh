#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. All Rights Reserved
python ./habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/gala_kinematic_ddppo.yaml --run-type train
