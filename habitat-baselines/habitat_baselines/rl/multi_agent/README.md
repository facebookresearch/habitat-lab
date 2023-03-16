# Running Training

* Running population play training: `python habitat_baselines/run.py --config-name=rearrange/rl_hierarchical.yaml habitat_baselines.evaluate=False habitat_baselines/rl/policy=hl_neural habitat_baselines/rl/policy/hierarchical_policy/defined_skills=oracle_skills_ma habitat_baselines/rl/agent=pop_play benchmark/rearrange=multi_agent_tidy_house.yaml`
