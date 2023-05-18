
```
HYDRA_FULL_ERROR=1 HABITAT_ENV_DEBUG=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  python habitat-baselines/habitat_baselines/run.py -m \
  --config-name experiments_hab3/socialnav_human_robot.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=20 \
  habitat_baselines.evaluate=False
```
