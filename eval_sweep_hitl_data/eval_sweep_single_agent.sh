#!/bin/bash

export datasets=("microtrain_eval_small_1scene_2objs_30epi_45degree" "microtrain_eval_medium_1scene_2objs_30epi_45degree" "microtrain_eval_large_1scene_2objs_30epi_45degree")

export dataset="microtrain_eval_small_1scene_2objs_30epi_45degree"
export dataset="microtrain_eval_medium_1scene_2objs_30epi_45degree"
export dataset="microtrain_eval_large_1scene_2objs_30epi_45degree"

MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python habitat-baselines/habitat_baselines/run.py -m --config-name experiments_hab3/single_agent_kinematic_oracle_humanoid_fp.yaml \
    habitat.simulator.kinematic_mode=True habitat.simulator.step_physics=False \
    habitat.seed=101,102,103 habitat_baselines.eval.save_summary_data=True habitat_baselines.writer_type=tb  \
    habitat_baselines.evaluate=True \
    habitat_baselines.num_environments=1 \
    habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/"${dataset}".json.gz \
    habitat.task.actions.oracle_nav_action.lin_speed=100 \
    habitat.task.actions.oracle_nav_action.ang_speed=40 \
    habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.plan_idx=3 \
    habitat.task.task_spec=rearrange_easy_fp habitat.task.pddl_domain_def=fp \
    hydra/launcher=aws_submitit_habitat_eval &


# multirun/2023-06-23/23-26-05
# multirun/2023-06-23/23-27-02
# multirun/2023-06-23/23-27-48
