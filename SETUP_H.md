# Setup for Humanoid

This has been tested both locally on MacOS and on Devfair
## Installation

1. **Preparing conda env**

   Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
   ```bash
   # We require python>=3.7 and cmake>=3.10
   conda create -n habitat python=3.7 cmake=3.14.0
   conda activate habitat
   ```

1. **conda install habitat-sim**
   - To install habitat-sim with bullet physics
      ```
      conda install habitat-sim withbullet -c conda-forge -c aihabitat
      ```
      See Habitat-Sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

1. **pip install habitat-lab mypy_cifix_03_h version**.

      ```bash
      git clone --branch mypy_cifix_03_h https://github.com/facebookresearch/habitat-lab.git
      cd habitat-lab
      pip install -e habitat-lab  # install habitat_lab
      ```
1. **Install habitat-baselines**.

    The command above will install only core of Habitat-Lab. To include habitat_baselines along with all additional requirements, use the command below after installing habitat-lab:

      ```bash
      pip install -e habitat-baselines  # install habitat_baselines
      ```


1. **Install Fairmotion inside data**
   ```bash
    cd data
    git clone https://github.com/facebookresearch/fairmotion.git
    cd fairmotion
    pip install -e .
    ```

## Download files

1. **Download environments and assets**

For now, it is enough with running:
```bash
cd ..
python examples/example.py
```

1. **Download human model data**
Unzip `human_sim_data.zip` inside `data` folder.

## Testing

You can test a hierarchical pick and place policy using the following command:

```bash
HABITAT_ENV_DEBUG=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -u habitat-baselines/habitat_baselines/run.py \
--exp-config habitat-baselines/habitat_baselines/config/rearrange/tp_srl_oracle_nav_human.yaml \
--run-type eval 
```

It will generate a video of the humanoid picking and placing an object.

## Interactive Testing
You can also test an interactive version by running

```bash
HABITAT_ENV_DEBUG=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python examples/interactive_play_human.py --never-end --cfg  habitat-lab/habitat/config/benchmark/rearrange/rearrange_human.yaml
```

Press `W, A, S ,D` to change the navigation target. Press `I,J,Y,T` to control the location of the arm.
