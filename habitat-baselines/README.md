baselines
==============================
### Installation

The `habitat_baselines` sub-package is NOT included upon installation by default. To install `habitat_baselines`, use the following command instead:
```bash
pip install -e habitat-lab
pip install -e habitat-baselines
```
This will also install additional requirements for each sub-module in `habitat_baselines/`, which are specified in `requirements.txt` files located in the sub-module directory.


### Reinforcement Learning (RL)

**Proximal Policy Optimization (PPO)**

**paper**: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

**code**: The PPO implementation is based on
[pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

**dependencies**: A recent version of pytorch, for installing refer to [pytorch.org](https://pytorch.org/)

For training on sample data please follow steps in the repository README. You should download the sample [test scene data](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip), extract it under the main repo (`habitat-lab/`, extraction will create a data folder at `habitat-lab/data`) and run the below training command.

**train**:
```bash
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav_example.yaml
```

You can reduce training time by changing the trainer from the default implement to [VER](rl/ver/README.md) by
setting `trainer_name` to `"ver"` in either the config or via the command line.

```bash
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav_example.yaml \
  habitat_baselines.trainer_name=ver
```

**test**:
```bash
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav_example.yaml \
  habitat_baselines.evaluate=True
```

We also provide trained RGB, RGBD, and Depth PPO  models for MatterPort3D and Gibson.
To use them download pre-trained pytorch models from [link](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/habitat_baselines_v2.zip) and unzip and specify model path [here](agents/ppo_agents.py#L151).

The `habitat_baselines/config/pointnav/ppo_pointnav.yaml` config has better hyperparameters for large scale training and loads the [Gibson PointGoal Navigation Dataset](/README.md#datasets) instead of the test scenes.
Change the `/benchmark/nav/pointnav: pointnav_gibson` in `habitat_baselines/config/pointnav/ppo_pointnav.yaml` to `/benchmark/nav/pointnav: pointnav_mp3d` in the defaults list for training on [MatterPort3D PointGoal Navigation Dataset](/README.md#datasets).

### Hierarchical Reinforcement Learning (HRL)

We provide a two-layer hierarchical policy class, consisting of a low-level skill that moves the robot, and a high-level policy that reasons about which low-level skill to use in the current state. This can be especially powerful in long-horizon mobile manipulation tasks, like those introduced in [Habitat2.0](https://arxiv.org/abs/2106.14405). Both the low- and high- level can be either learned or an oracle. For oracle high-level we use [PDDL](https://planning.wiki/guide/whatis/pddl), and for oracle low-level we use instantaneous transitions, with the environment set to the final desired state. Additionally, for navigation, we provide an oracle navigation skill that uses A-star and the map of the environment to move the robot to its goal.

To run the following examples, you need the [ReplicaCAD dataset](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replicacad).

To train a high-level policy, while using pre-learned low-level skills (SRL baseline from [Habitat2.0](https://arxiv.org/abs/2106.14405)), you can run:

```bash
python -u -m habitat_baselines.run \
  --config-name=rearrange/rl_hierarchical.yaml
```
To run a rearrangement episode with oracle low-level skills and a fixed task planner, run:

```bash
python -u -m habitat_baselines.run \
  --config-name=rearrange/rl_hierarchical.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines/rl/policy=hl_fixed \
  habitat_baselines/rl/policy/hierarchical_policy/defined_skills=oracle_skills
```

To change the task (like set table) that you train your skills on, you can change the line `/habitat/task/rearrange: rearrange_easy` to `/habitat/task/rearrange: set_table` in the defaults of your config.

# Habitat-3.0 Multi-Agent Training
First download the necessary data with `python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab3_bench_assets`.

## Social Navigation

To run multi-agent training with a Spot robot's policy being a low-level navigation policy and a humanoid's policy being a fixed (non-trainable) policy that navigates a sequence of navigation targets.
- `python habitat_baselines/run.py --config-name=social_nav/social_nav.yaml`

For evaluating the trained Spot robot's policy
- `python habitat_baselines/run.py --config-name=social_nav/social_nav.yaml habitat_baselines.evaluate=True habitat_baselines.eval_ckpt_path_dir=/checkpoints/latest.pth habitat_baselines.eval.should_load_ckpt=True`

## Social Rearrangement

To run multi-agent training with a Spot robot and humanoid on the social rearrangement task.
- Learn-Single: `python habitat_baselines/run.py --config-name=social_rearrange/pop_play.yaml`
- Learn-Pop with 8 humanoid policies during training: `python habitat_baselines/run.py --config-name=social_rearrange/pop_play.yaml habitat_baselines.rl.agent.num_pool_agents_per_type=[1,8]`
- Plan-Pop-4: `python habitat_baselines/run.py --config-name=social_rearrange/plan_pop.yaml habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=4`. To run Plan-Pop-p for other `p` values, set `habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx`.

For zero-shot evaluate against the unseen agent population:
- With planner-based collaborators: `python habitat_baselines/run.py --config-name=social_rearrange/pop_play.yaml habitat_baselines.evaluate=True habitat_baselines.eval_ckpt_path_dir=PATH_TO_CKPT.pth +habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.select_random_goal=False +habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=1` change `plan_idx` to be 1, 2, 3, or 4 to evaluate against the other 4 planner agents.
