#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: run this script from habitat-lab/ directory
# TO PLOT RESULTS SEE RUN `python scripts/hab2_bench/plot_bench.py`
mkdir -p data/profile
NUM_STEPS=200
set -e

export OMP_NUM_THREADS=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

USE_ORACLE_ACTION=
NO_SKINNING=

#number of processes
# shellcheck disable=SC2043
for j in 1
do
  #number of trials
  for i in {0..1}
  do

    #TODO: different configs for different agent pairs. Can we make a single high-level config

    #Single agent robot
    # python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/rearrange_easy_human_and_spot.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "human_and_spot_$i"

    # python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/rearrange_easy_human_and_spot.yaml --n-steps 1 --n-procs 1 --out-name test
    # python scripts/hab3_bench/hab3_benchmark.py --cfg hab3_bench/spot_humanoid_oracle.yaml --n-steps 1 --n-procs 1 --out-name test
    #multi-agent robots

    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_spot_vel.yaml --n-steps 30 --n-procs 1 --out-name test  

    #multi-agent robot, human (no skinning)

    #multi-agent robot, human (+skinning)
    # python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_humanoid_vel.yaml --n-steps 1 --n-procs 1 --out-name test --render
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_humanoid_vel.yaml --n-steps 300 --n-procs 1 --out-name test 

    #multi-agent robot, human (+skinning) + path actions
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_humanoid_oracle.yaml --n-steps 300 --n-procs 1 --out-name test

    #stretch features:
    #HSSD vs ReplicaCAD
    #pick/place vs nav (requires calling skills)
    #joints vs base control
    #robot continuous control modes (backup vs no backup)

  done
done