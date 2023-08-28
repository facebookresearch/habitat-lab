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
#NOTE: this creates a new URDF with no accompanying ao_config to avoid skinning
cp data/humanoids/humanoid_data/female2_0.urdf data/humanoids/humanoid_data/female2_0_no_skinning.urdf
NO_SKINNING="habitat.simulator.agents.agent_1.articulated_agent_urdf='data/humanoids/humanoid_data/female2_0_no_skinning.urdf'"

DATA_LARGE="habitat.dataset.datapath="


OBJ3="habitat.task.task_spec=rearrange_easy_fp_3obj"
OBJ5="habitat.task.task_spec=rearrange_easy_fp_5obj"




# number of processes
# shellcheck disable=SC2043
for j in 1
do
  #number of trials
  for i in {1..10}
  do

    #TODO: different configs for different agent pairs. Can we make a single high-level config

    #Single agent robot
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_oracle.yaml --n-steps 300 --n-procs 1 --out-name "robot_oracle_$i"

    #Single agent robot - 3 objects
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_oracle.yaml --n-steps 300 --n-procs 1 --out-name "robot_oracle_3obj_$i" "$OBJ3"

    #Single agent robot - 5 objects
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_oracle.yaml --n-steps 300 --n-procs 1 --out-name "robot_oracle_5obj_$i" "$OBJ5"

    #Single agent robot - Large scene
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_oracle.yaml --n-steps 300 --n-procs 1 --out-name "robot_oracle_large_$i" "$DATA_LARGE"


    # Humanoid oracle
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/humanoid_oracle.yaml --n-steps 300 --n-procs 1 --out-name "human_oracle_$i"


    #multi-agent robots
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_spot_vel.yaml --n-steps 300 --n-procs 1 --out-name "robots_vel_$i"

    #multi-agent robot, human (no skinning)
    #python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_humanoid_vel.yaml --n-steps 300 --n-procs 1 --out-name "robot_human_vel_noskin_$i" "$NO_SKINNING"

    #multi-agent robot, human (+skinning)
    # python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_humanoid_vel.yaml --n-steps 1 --n-procs 1 --out-name test --render
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_humanoid_vel.yaml --n-steps 300 --n-procs 1 --out-name "robot_human_vel_$i"

    #multi-agent robot, human (+skinning) + path actions
    python scripts/hab3_bench/hab3_benchmark.py --cfg benchmark/rearrange/hab3_bench/spot_humanoid_oracle.yaml --n-steps 300 --n-procs 1 --out-name "robot_human_oracle_$i"

    #stretch features:
    #HSSD vs ReplicaCAD
    #pick/place vs nav (requires calling skills)
    #joints vs base control
    #robot continuous control modes (backup vs no backup)

  done
done
