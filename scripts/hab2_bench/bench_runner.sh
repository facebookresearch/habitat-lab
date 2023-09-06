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

NO_SLEEP="habitat.simulator.auto_sleep=False"
NO_CONCUR="habitat.simulator.concur_render=False"
#NO_PHYSICS=("habitat.simulator.step_physics" False) #disables simulation step and articulated agent update
#NO_ROBOT_UPDATE=("habitat.simulator.update_articulated_agent" False) #only disables articulated agent update

#number of processes
# shellcheck disable=SC2043
for j in 1 16
do
  #number of trials
  for i in {0..10}
  do
    ##################################
    # IDLE 4 sensors (arm + head, RGBD)
    ##################################
    # # Ours
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "all_$i"

    # # Ours (-Concur Render)
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "noconcur_$i" "$NO_CONCUR"

    # # Ours (-Auto sleep)
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "nosleep_$i" "$NO_SLEEP"

    # # Ours (RENDER_ONLY)
    # python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "render_$i" "${NO_PHYSICS[@]}" "${NO_ROBOT_UPDATE[@]}"

    ##################################
    # IDLE 1 sensor (head RGB)
    ##################################

    # # Ours
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "all_$i"

    # # Ours (-Concur Render)
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "noconcur_$i" "$NO_CONCUR"

    # # Ours (-Auto sleep)
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "nosleep_$i" "$NO_SLEEP"

    # # Ours (RENDER_ONLY)
    # python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "render_$i" "${NO_PHYSICS[@]}" "${NO_ROBOT_UPDATE[@]}"

    ##################################
    # INTERACT 4 sensors (arm + head, RGBD)
    ##################################

    # # Ours
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/interact.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "all_$i"

    # # Ours (-Concur Render)
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/interact.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "noconcur_$i" "$NO_CONCUR"

    # # Ours (-Auto sleep)
    python scripts/hab2_bench/hab2_benchmark.py --cfg benchmark/rearrange/demo/interact.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "nosleep_$i" "$NO_SLEEP"

  done
done
