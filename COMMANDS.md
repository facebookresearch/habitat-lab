### Debugging
```
HYDRA_FULL_ERROR=1 HABITAT_ENV_DEBUG=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  python habitat-baselines/habitat_baselines/run.py -m \
  --config-name experiments_hab3/socialnav_human_robot.yaml \
  habitat_baselines.num_environments=1
```

### Evaluating oracle policy
```
HYDRA_FULL_ERROR=1 HABITAT_ENV_DEBUG=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  python habitat-baselines/habitat_baselines/run.py -m \
  --config-name experiments_hab3/socialnav_human_robot_oracle.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval.should_load_ckpt=False \
  habitat_baselines.test_episode_count=10
```

### Training
Training parameters in `submitit_habitat.yaml`
```
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  python habitat-baselines/habitat_baselines/run.py -m \
  --config-name experiments_hab3/socialnav_human_robot.yaml \
  habitat_baselines.num_environments=1 \
  hydra/launcher=submitit_habitat
```

### Evaluation
```
habitat_baselines.evaluate=True
habitat_baselines.eval_ckpt_path_dir=
```
