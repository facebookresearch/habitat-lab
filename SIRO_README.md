# SIRo Project

Project-specific README for SIRo.

# Installation

1. Clone the Habitat-lab [SIRo branch](https://github.com/facebookresearch/habitat-lab/tree/SIRo).
1. Install Habitat-lab using [instructions](https://github.com/facebookresearch/habitat-lab/tree/SIRo#installation).
1. Install Habitat-sim `main` branch.
    * [Build from source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md), or install the [conda nightly build](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages).
        * Be sure to include Bullet physics, e.g. `python setup.py install --bullet`.
    * Anecdotally, building from source is working more reliably (versus the conda nightly build).
    * If you build from source, configure `PYTHONPATH` and ensure that Python `import habitat_sim` imports your locally-built version of Habitat-sim.
    * Keep an eye on relevant [commits to main](https://github.com/facebookresearch/habitat-sim/commits/main) to help you decide when to update/rebuild Habitat-sim.
    * If you encounter the issue about cmake when building from source using Mac `missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun`, you can solve this by first install the tools `xcode-select --install`.
1. Download humanoid data.
    * From the Habitat-lab root directory, `python -m habitat_sim.utils.datasets_download --uids humanoid_data  --data-path data/`
    * Manually download `walking_motion_processed.pkl` from [this Slack thread](https://cvmlp.slack.com/archives/C0460NTKM4G/p1678403985106999?thread_ts=1678402520.813389&cid=C0460NTKM4G) to `data/humanoids/humanoid_data/walking_motion_processed.pkl`
1. Download other required datasets:
    * `python -m habitat_sim.utils.datasets_download --uids ycb hab_fetch hab_spot_arm replica_cad_dataset rearrange_pick_dataset_v0 rearrange_dataset_v1 --data-path data/`

# Sandbox Tool

see [Sandbox Tool Readme](./examples/siro_sandbox/README.md)

# Training

## Multi-Agent

### Fetch-Fetch
Fetch-Fetch in ReplicaCAD multi-agent training, single GPU. From `habitat-lab` directory:
```
HABITAT_SIM_LOG=warning:physics,metadata=quiet MAGNUM_LOG=warning python habitat-baselines/habitat_baselines/run.py -m --config-name experiments_hab3/pop_play_kinematic_oracle.yaml hydra/output=path
```
This will create a directory `outputs/pop-play/<date>/<time>/0` and store data like checkpoints and logs into that folder. If you would like to edit the path where your run data is stored, you can edit `config/hydra/output/path.yaml` to take other paths.

### Fetch-Humanoid
To run a Fetch-Humanoid Policy on ReplicaCAD, single GPU, you will need to run:
```
HABITAT_SIM_LOG=warning:physics,metadata=quiet MAGNUM_LOG=warning python habitat-baselines/habitat_baselines/run.py -m --config-name experiments_hab3/pop_play_kinematic_oracle_humanoid.yaml hydra/output=path
```
Note that the default value for population here is [1,1], meaning that we will be training a single policy for each agent. The argument `rl.agent.num_pool_agents_per_type` can be changed to [1,8] for population based training, where the humanoid is samples from 8 policies.


# Eval

To run evaluation, run, from `habitat-lab` directory:

```
sh eval_sweep.sh
```

You will be prompted to enter a directory `$SWEEP_SUBDIR` name where the checkpoints and config files are saved (normally in the format `name/yyyy-dd-mm/hh-mm-ss`). The script will generate videos of evaluation at `$SWEEP_SUBDIR/0/video`.

## Demo Fetch-Human Fixed Planner

You can also run a Fetch-Humanoid where both work with a Fixed Planner, using:

```
python  habitat-baselines/habitat_baselines/run.py -m  habitat_baselines.evaluate=True habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=False  --config-name experiments_hab3/rearrange_fetch_human_planner.yaml
```

# Spot robot

## Testing the Spot

To run Spot in FP (`pop_play_kinematic_oracle_spot_fp.yaml`), please follows the following instruction

Dataset
1. Download the asset from the [Google Drive](https://drive.google.com/file/d/1-utUMfUbbzg_zUE5GcGNdk1UEK6lSbXe/view?usp=sharing)
2. Unzip the file and put it in `habitat_lab/habitat-lab/data/fpss`
3. Download the amazon and google objects from the [Google Drive](https://drive.google.com/drive/folders/1x6i3sDYheCWoi59lv27ZyPG4Ii2GhEZB?usp=share_link)
4. Put them in `habitat_lab/habitat-lab/data/objects`
5. Download the episode (there is only single scene) from the [Google Drive](https://drive.google.com/file/d/1Guxn6v2SC5kAtouwMs1DkBJMK3WW0die/view?usp=sharing)
6. Unzip it and put them in `habitat-lab/habitat-lab/data/datasets/floorplanner/`

Program
7. Assume you have already setup the habitat, then `cd habitat-lab/`
8. `srun -v --gpus-per-node=1 --partition=siro --time=1:00:00 --cpus-per-task 1 python -u habitat-baselines/habitat_baselines/run.py --config-name=experiments_hab3/pop_play_kinematic_oracle_spot_fp.yaml habitat_baselines.num_environments=1`

or for running HRL fix policy
`python habitat-baselines/habitat_baselines/run.py --config-name=rearrange/rl_hierarchical_oracle_nav_spot_fp.yaml habitat_baselines.evaluate=True habitat.simulator.kinematic_mode=True habitat.simulator.step_physics=False habitat.task.measurements.force_terminate.max_accum_force=-1.0 habitat.task.measurements.force_terminate.max_instant_force=-1.0 habitat_baselines.num_environments=1 habitat_baselines/rl/policy/hierarchical_policy/defined_skills@habitat_baselines.rl.policy.main_agent.hierarchical_policy.defined_skills=oracle_skills`

TODO
1. Generate more scenes
2. Fix Spot robot navmesh issue
