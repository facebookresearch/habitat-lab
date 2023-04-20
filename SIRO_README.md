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
1. Download humanoid data.
    * From the Habitat-lab root directory, `python -m habitat_sim.utils.datasets_download --uids humanoid_data  --data-path data/`
    * Manually download `walking_motion_processed.pkl` from [this Slack thread](https://cvmlp.slack.com/archives/C0460NTKM4G/p1678403985106999?thread_ts=1678402520.813389&cid=C0460NTKM4G) to `data/humanoids/humanoid_data/walking_motion_processed.pkl`
1. Download other required datasets:
    * `python -m habitat_sim.utils.datasets_download --uids ycb hab_fetch hab_spot_arm replica_cad_dataset rearrange_pick_dataset_v0 rearrange_dataset_v1 --data-path data/`

# Sandbox Tool

see [Sandbox Tool Readme](./examples/siro_sandbox/README.md)

# Training

## Multi-Agent
Fetch-Fetch in ReplicaCAD multi-agent training, single GPU. From `habitat-lab` directory:
```
HABITAT_SIM_LOG=warning:physics,metadata=quiet MAGNUM_LOG=warning python habitat-baselines/habitat_baselines/run.py -m --config-name experiments_hab3/pop_play_kinematic_oracle.yaml hydra/output=path
```
This will create a directory `outputs/pop-play/<date>/<time>/0` and store data like checkpoints and logs into that folder. If you would like to edit the path where your run data is stored, you can edit `config/hydra/output/path.yaml` to take other paths.

Fetch-Humanoid Training
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

# Spot robot

## Testing the Spot

todo: terminal commands, etc.
