#!/bin/bash

# TO PLOT RESULTS SEE RUN `python scripts/plot_bench.py`
mkdir -p data/profile
NUM_STEPS=200
set -e

export OMP_NUM_THREADS=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

NO_SLEEP=("SIMULATOR.AUTO_SLEEP" False)
NO_CONCUR=("SIMULATOR.CONCUR_RENDER" False)
NO_PHYSICS=("SIMULATOR.STEP_PHYSICS" False) #also disables robot update
NO_ROBOT_UPDATE=("SIMULATOR.UPDATE_ROBOT" False)
#NO_OBJS=("SIMULATOR.LOAD_ART_OBJS" False "SIMULATOR.LOAD_OBJS" False)

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
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "all_$i"

    # # Ours (-Concur Render)
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "noconcur_$i" "${NO_CONCUR[@]}"

    # # Ours (-Auto sleep)
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "nosleep_$i" "${NO_SLEEP[@]}"

    # # Ours (RENDER_ONLY)
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "render_$i" "${NO_PHYSICS[@]}" "${NO_ROBOT_UPDATE[@]}"

    ##################################
    # IDLE 1 sensor (head RGB)
    ##################################

    # # Ours
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "all_$i"

    # # Ours (-Concur Render)
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "noconcur_$i" "${NO_CONCUR[@]}"

    # # Ours (-Auto sleep)
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "nosleep_$i" "${NO_SLEEP[@]}"

    # # Ours (RENDER_ONLY)
    python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/idle_single_camera.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "render_$i" "${NO_PHYSICS[@]}" "${NO_ROBOT_UPDATE[@]}"

    ##################################
    # INTERACT 4 sensors (arm + head, RGBD)
    ##################################
    #TODO:

    # # Ours
    #python scripts/hab2_benchmark.py --cfg configs/tasks/rearrange/benchmark/interact.yaml --n-steps "$NUM_STEPS" --n-procs "$j" --out-name "all_$i"

  done
done
