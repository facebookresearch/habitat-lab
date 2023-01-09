[![CircleCI](https://circleci.com/gh/facebookresearch/habitat-lab.svg?style=shield)](https://circleci.com/gh/facebookresearch/habitat-lab)
[![codecov](https://codecov.io/gh/facebookresearch/habitat-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/facebookresearch/habitat-lab)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/habitat-lab/blob/main/LICENSE)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/facebookresearch/habitat-lab)](https://github.com/facebookresearch/habitat-lab/releases/latest)
[![Supports Habitat_Sim](https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational&link=https://github.com/facebookresearch/habitat-sim)](https://github.com/facebookresearch/habitat-sim)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![Twitter Follow](https://img.shields.io/twitter/follow/ai_habitat?style=social)](https://twitter.com/ai_habitat)

Habitat-Lab
==============================

Habitat-Lab is a modular high-level library for end-to-end development in embodied AI --
defining embodied AI tasks (e.g. navigation, rearrangement, instruction following, question answering), configuring embodied agents (physical form, sensors, capabilities), training these agents (via imitation or reinforcement learning, or no learning at all as in SensePlanAct pipelines), and benchmarking their performance on the defined tasks using standard metrics.

Habitat-Lab uses [`Habitat-Sim`](https://github.com/facebookresearch/habitat-sim) as the core simulator. For documentation refer [here](https://aihabitat.org/docs/habitat-lab/).

[![Habitat Demo](https://img.shields.io/static/v1?label=WebGL&message=Try%20AI%20Habitat%20In%20Your%20Browser%20&color=blue&logo=webgl&labelColor=%23990000&style=for-the-badge&link=https://aihabitat.org/demo)](https://aihabitat.org/demo)
<p align="center">
  <img src="res/img/habitat_compressed.gif"  height="400">
</p>

---

## Table of contents
   1. [Citing Habitat](#citing-habitat)
   1. [Installation](#installation)
   1. [Testing](#testing)
   1. [Documentation](#documentation)
   1. [Docker Setup](#docker-setup)
   1. [Datasets](#datasets)
   1. [Baselines](#baselines)
   1. [License](#license)


## Citing Habitat
If you use the Habitat platform in your research, please cite the [Habitat 1.0](https://arxiv.org/abs/1904.01201) and [Habitat 2.0](https://arxiv.org/abs/2106.14405) papers:

```
@inproceedings{szot2021habitat,
  title     =     {Habitat 2.0: Training Home Assistants to Rearrange their Habitat},
  author    =     {Andrew Szot and Alex Clegg and Eric Undersander and Erik Wijmans and Yili Zhao and John Turner and Noah Maestre and Mustafa Mukadam and Devendra Chaplot and Oleksandr Maksymets and Aaron Gokaslan and Vladimir Vondrus and Sameer Dharur and Franziska Meier and Wojciech Galuba and Angel Chang and Zsolt Kira and Vladlen Koltun and Jitendra Malik and Manolis Savva and Dhruv Batra},
  booktitle =     {Advances in Neural Information Processing Systems (NeurIPS)},
  year      =     {2021}
}

@inproceedings{habitat19iccv,
  title     =     {Habitat: {A} {P}latform for {E}mbodied {AI} {R}esearch},
  author    =     {Manolis Savva and Abhishek Kadian and Oleksandr Maksymets and Yili Zhao and Erik Wijmans and Bhavana Jain and Julian Straub and Jia Liu and Vladlen Koltun and Jitendra Malik and Devi Parikh and Dhruv Batra},
  booktitle =     {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      =     {2019}
}
```

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

1. **pip install habitat-lab stable version**.

      ```bash
      git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
      cd habitat-lab
      pip install -e habitat-lab  # install habitat_lab
      ```
1. **Install habitat-baselines**.

    The command above will install only core of Habitat-Lab. To include habitat_baselines along with all additional requirements, use the command below after installing habitat-lab:

      ```bash
      pip install -e habitat-baselines  # install habitat_baselines
      ```

## Testing

1. Let's download some 3D assets using Habitat-Sim's python data download utility:
   - Download (testing) 3D scenes:
      ```bash
      python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
      ```
      Note that these testing scenes do not provide semantic annotations.

   - Download point-goal navigation episodes for the test scenes:
      ```bash
      python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
      ```

1. **Non-interactive testing**: Test the PointNav task: Run the example pointnav script
    ```bash
    python examples/example_pointnav.py
    ```

    which instantiates a PointNav agent in the testing scenes and episodes. The agent takes random actions and you should see something like:

    ```bash
    [16:11:11:718584]:[Sim] Simulator.cpp(205)::reconfigure : CreateSceneInstance success == true for active scene name : data/scene_datasets/habitat-test-scenes/skokloster-castle.glb  with renderer.
    2022-08-13 16:26:45,068 Initializing task Nav-v0
    Environment creation successful
    Agent stepping around inside environment.
    Episode finished after 5 steps.
    ```

1. **Non-interactive testing**: Test the Pick task: Run the example pick task script
    <!--- Please, update `examples/example.py` if you update example. -->
    ```bash
    python examples/example.py
    ```

    which uses [`habitat-lab/habitat/config/benchmark/rearrange/pick.yaml`](habitat-lab/habitat/config/benchmark/rearrange/pick.yaml) for configuration of task and agent.

    ```python
    import habitat

    # Load embodied AI task (RearrangePick) and a pre-specified virtual robot
    env = habitat.Env(config=habitat.get_config("benchmark/rearrange/pick.yaml"))
    observations = env.reset()

    # Step through environment with random actions
    while not env.episode_over:
        observations = env.step(env.action_space.sample())
    ```

    This script instantiates a Pick agent in ReplicaCAD scenes. The agent takes random actions and you should see something like:
    ```bash
      Agent acting inside environment.
      Renderer: AMD Radeon Pro 5500M OpenGL Engine by ATI Technologies Inc.
      Episode finished after 200 steps.
    ```

    See [`examples/register_new_sensors_and_measures.py`](examples/register_new_sensors_and_measures.py) for an example of how to extend habitat-lab from _outside_ the source code.


1. **Interactive testing**: Using you keyboard and mouse to control a Fetch robot in a ReplicaCAD environment:
    ```bash
    # Pygame for interactive visualization, pybullet for inverse kinematics
    pip install pygame==2.0.1 pybullet==3.0.4

    # Interactive play script
    python examples/interactive_play.py --never-end --add-ik
    ```

   Use I/J/K/L keys to move the robot base forward/left/backward/right and W/A/S/D to move the arm end-effector forward/left/backward/right and E/Q to move the arm up/down. The arm can be difficult to control via end-effector control. More details in documentation. Try to move the base and the arm to touch the red bowl on the table. Have fun!


## Documentation

Browse the online [Habitat-Lab documentation](https://aihabitat.org/docs/habitat-lab/index.html) and the extensive [tutorial on how to train your agents with Habitat](https://aihabitat.org/tutorial/2020/). For Habitat 2.0, use this [quickstart guide](https://aihabitat.org/docs/habitat2/).


## Docker Setup
We provide docker containers for Habitat, updated approximately once per year for the [Habitat Challenge](https://github.com/facebookresearch/habitat-challenge). This works on machines with an NVIDIA GPU and requires users to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). To setup the habitat stack using docker follow the below steps:

1. Pull the habitat docker image: `docker pull fairembodied/habitat-challenge:testing_2022_habitat_base_docker`

1. Start an interactive bash session inside the habitat docker: `docker run --runtime=nvidia -it fairembodied/habitat-challenge:testing_2022_habitat_base_docker`

1. Activate the habitat conda environment: `conda init; source ~/.bashrc; source activate habitat`

1. Run the testing scripts as above: `cd habitat-lab; python examples/example_pointnav.py`. This should print out an output like:
    ```bash
    Agent acting inside environment.
    Episode finished after 200 steps.
    ```

### Questions?
Can't find the answer to your question? Try asking the developers and community on our [Discussions forum](https://github.com/facebookresearch/habitat-lab/discussions).

## Datasets

[Common task and episode datasets used with Habitat-Lab](DATASETS.md).

## Baselines
Habitat-Lab includes reinforcement learning (via PPO) and classical SLAM based baselines. For running PPO training on sample data and more details refer [habitat_baselines/README.md](habitat-baselines/habitat_baselines/README.md).

## ROS-X-Habitat
ROS-X-Habitat (https://github.com/ericchen321/ros_x_habitat) is a framework that bridges the AI Habitat platform (Habitat Lab + Habitat Sim) with other robotics resources via ROS. Compared with Habitat-PyRobot, ROS-X-Habitat places emphasis on 1) leveraging Habitat Sim v2's physics-based simulation capability and 2) allowing roboticists to access simulation assets from ROS. The work has also been made public as a [paper](https://arxiv.org/abs/2109.07703).

Note that ROS-X-Habitat was developed, and is maintained by the Lab for Computational Intelligence at UBC; it has not yet been officially supported by the Habitat Lab team. Please refer to the framework's repository for docs and discussions.


## License
Habitat-Lab is MIT licensed. See the [LICENSE file](/LICENSE) for details.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.

- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Gibson based task datasets, the code for generating such datasets, and trained models are distributed with [Gibson Terms of Use](https://storage.googleapis.com/gibson_material/Agreement%20GDS%2006-04-18.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
