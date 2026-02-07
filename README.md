[![codecov](https://codecov.io/gh/facebookresearch/habitat-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/facebookresearch/habitat-lab)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/habitat-lab/blob/main/LICENSE)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/facebookresearch/habitat-lab)](https://github.com/facebookresearch/habitat-lab/releases/latest)
[![Supports Habitat_Sim](https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational&link=https://github.com/facebookresearch/habitat-sim)](https://github.com/facebookresearch/habitat-sim)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![Twitter Follow](https://img.shields.io/twitter/follow/ai_habitat?style=social)](https://twitter.com/ai_habitat)

Deep RL: VLM-Guided Subgoal Planning for Indoor Navigation in Habitat-Lab
==============================

## Installation

1. **Preparing conda env**

   Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
   ```bash
   # We require python>=3.9 and cmake>=3.14
   conda create -n habitat python=3.9 cmake=3.14.0
   conda activate habitat
   ```

1. **conda install habitat-sim**
   - To install habitat-sim with bullet physics
      ```
      conda install habitat-sim withbullet -c conda-forge -c aihabitat
      ```
      Note, for newer features added after the most recent release, you may need to install `aihabitat-nightly`. See Habitat-Sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

1. **pip install habitat-lab stable version**.

      ```bash
      git clone https://github.com/meganmlee/HabitatLab_VLM_PPO.git
      cd habitat-lab
      pip install -e habitat-lab  # install habitat_lab
      ```
1. **Install habitat-baselines**.

    The command above will install only core of Habitat-Lab. To include habitat_baselines along with all additional requirements, use the command below after installing habitat-lab:

      ```bash
      pip install -e habitat-baselines  # install habitat_baselines
      ```

## Datasets

Install HM3D scenes by following [these instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d):

      python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_full

Install Objectnav dataset from here (Object goal navigation zip file for HM3DSem-v0.1). [Common task and episode datasets used with Habitat-Lab](DATASETS.md).

## Baselines

To train. Change what you need to:

      python habitat-baselines/habitat_baselines/run.py habitat_baselines.num_updates=25000 habitat_baselines.num_environments=16 --config-name="objectnav/ddppo_objectnav_hm3d.yaml"
      
To evaluate. Change what you need to:

      python habitat-baselines/habitat_baselines/run.py habitat_baselines.evaluate=True habitat_baselines.eval_ckpt_path_dir="data/new_checkpoints/ckpt.21.pth" habitat_baselines.checkpoint_folder="data/eval_temp" habitat_baselines.num_environments=16  habitat_baselines.test_episode_count=2000 --config-name="objectnav/ddppo_objectnav_hm3d.yaml"

Habitat-Lab includes reinforcement learning (via PPO) baselines. For running PPO training on sample data and more details refer [habitat_baselines/README.md](habitat-baselines/habitat_baselines/README.md).


## License
Habitat-Lab is MIT licensed. See the [LICENSE file](/LICENSE) for details.

Copyright (c) Meta Platforms, Inc. and affiliates.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.

- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Gibson based task datasets, the code for generating such datasets, and trained models are distributed with [Gibson Terms of Use](https://storage.googleapis.com/gibson_material/Agreement%20GDS%2006-04-18.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
