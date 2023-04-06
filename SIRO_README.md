# SIRo Project

Project-specific README for SIRo.

# Installation

1. Clone the Habitat-lab [SIRo branch](https://github.com/facebookresearch/habitat-lab/tree/SIRo).
1. Install Habitat-lab using [instructions](https://github.com/facebookresearch/habitat-lab/tree/SIRo#installation).
1. Install Habitat-sim `main` branch.
    * [Build from source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md), or install the [conda nightly build](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages).
    * If you build from source, configure `PYTHONPATH` and ensure that Python `import habitat_sim` imports your locally-built version of Habitat-sim.
    * Keep an eye on relevant [commits to main](https://github.com/facebookresearch/habitat-sim/commits/main) to help you decide when to update/rebuild Habitat-sim.
1. Download humanoid data.
    * From the Habitat-lab root directory, `python -m habitat_sim.utils.datasets_download --uids humanoid_data  --data-path data/`
    * Manually download `walking_motion_processed.pkl` from [this Slack thread](https://cvmlp.slack.com/archives/C0460NTKM4G/p1678403985106999?thread_ts=1678402520.813389&cid=C0460NTKM4G) to `data/humanoids/humanoid_data/walking_motion_processed.pkl`
2. Note other required datasets that should download automatically as needed when running Habitat-lab:
    * hab_ycb, hab_fetch, replica_cad_dataset, rearrange_pick_dataset_v0, rearrange_dataset_v1

# Sandbox Tool

see [Sandbox Tool Readme](./examples/siro_sandbox/README.md)

# Training

todo: terminal commands, etc.

# Eval

todo: terminal commands, etc.

# Spot robot

## Testing the Spot

todo: data download instructions, terminal commands, etc.
