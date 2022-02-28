[![CircleCI](https://circleci.com/gh/facebookresearch/habitat-lab.svg?style=shield)](https://circleci.com/gh/facebookresearch/habitat-lab)
[![codecov](https://codecov.io/gh/facebookresearch/habitat-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/facebookresearch/habitat-lab)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/habitat-lab/blob/main/LICENSE)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/facebookresearch/habitat-lab)](https://github.com/facebookresearch/habitat-lab/releases/latest)
[![Supports Habitat_Sim](https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational&link=https://github.com/facebookresearch/habitat-sim)](https://github.com/facebookresearch/habitat-sim)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![Slack Join](http://img.shields.io/static/v1?label=Join%20us%20on&message=%23habitat-dev&labelColor=%234A154B&logo=slack)](https://join.slack.com/t/ai-habitat/shared_invite/enQtNjY1MzM1NDE4MTk2LTZhMzdmYWMwODZlNjg5MjZiZjExOTBjOTg5MmRiZTVhOWQyNzk0OTMyN2E1ZTEzZTNjMWM0MjBkN2VhMjQxMDI)
[![Twitter Follow](https://img.shields.io/twitter/follow/ai_habitat?style=social)](https://twitter.com/ai_habitat)

Habitat Lab
==============================

Habitat Lab is a modular high-level library for end-to-end development in embodied AI –
defining embodied AI tasks (e.g. navigation, instruction following, question answering), configuring embodied agents (physical form, sensors, capabilities), training these agents (via imitation or reinforcement learning, or no learning at all as in classical SLAM), and benchmarking their performance on the defined tasks using standard metrics.

Habitat Lab currently uses [`Habitat-Sim`](https://github.com/facebookresearch/habitat-sim) as the core simulator, but is designed with a modular abstraction for the simulator backend to maintain compatibility over multiple simulators. For documentation refer [here](https://aihabitat.org/docs/habitat-lab/).

We also have a dev slack channel, please follow this [link](https://join.slack.com/t/ai-habitat/shared_invite/enQtNjY1MzM1NDE4MTk2LTZhMzdmYWMwODZlNjg5MjZiZjExOTBjOTg5MmRiZTVhOWQyNzk0OTMyN2E1ZTEzZTNjMWM0MjBkN2VhMjQxMDI) to get added to the channel. If you want to contribute PRs or face issues with habitat please reach out to us either through github issues or slack channel.

[![Habitat Demo](https://img.shields.io/static/v1?label=WebGL&message=Try%20AI%20Habitat%20In%20Your%20Browser%20&color=blue&logo=webgl&labelColor=%23990000&style=for-the-badge&link=https://aihabitat.org/demo)](https://aihabitat.org/demo)
<p align="center">
  <img src="res/img/habitat_compressed.gif"  height="400">
</p>

---

## Table of contents
   1. [Motivation](#motivation)
   1. [Citing Habitat](#citing-habitat)
   1. [Installation](#installation)
   1. [Example](#example)
   1. [Documentation](#documentation)
   1. [Docker Setup](#docker-setup)
   1. [Details](#details)
   1. [Data](#data)
   1. [Baselines](#baselines)
   1. [License](#license)
   1. [Acknowledgments](#acknowledgments)
   1. [References](#references)

## Motivation
While there has been significant progress in the vision and language communities thanks to recent advances in deep representations, we believe there is a growing disconnect between ‘internet AI’ and embodied AI. The focus of the former is pattern recognition in images, videos, and text on datasets typically curated from the internet. The focus of the latter is to enable action by an embodied agent in an environment (e.g. a robot). This brings to the forefront issues of active perception, long-term planning, learning from interaction, and holding a dialog grounded in an environment.

To this end, we aim to standardize the entire ‘software stack’ for training embodied agents – scanning the world and creating highly photorealistic 3D assets, developing the next generation of highly efficient and parallelizable simulators, specifying embodied AI tasks that enable us to benchmark scientific progress, and releasing modular high-level libraries to train and deploy embodied agents.

## Citing Habitat
If you use the Habitat platform in your research, please cite the [Habitat](https://arxiv.org/abs/1904.01201) and [Habitat 2.0](https://arxiv.org/abs/2106.14405) papers:

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

1. Clone a stable version from the github repository and install habitat-lab using the commands below. Note that python>=3.7 is required for working with habitat-lab. All the development and testing was done using python3.7. Please use 3.7 to avoid possible issues.

    ```bash
    git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e .
    ```

    The command above will install only core of Habitat Lab. To include habitat_baselines along with all additional requirements, use the command below instead:

    ```bash
    git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -r requirements.txt
    python setup.py develop --all # install habitat and habitat_baselines
    ```

2. Install `habitat-sim`:

      For a machine with an attached display,

      ```bash
      conda install habitat-sim withbullet -c conda-forge -c aihabitat
      ```

      For a machine with multiple GPUs or without an attached display (i.e. a cluster),

      ```bash
       conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
      ```

      See habitat-sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.
      MacOS does *not* work with `headless` so exclude that argument if on MacOS.


3. Run the example script `python examples/example.py ` which in the end should print out number of steps agent took inside an environment (eg: `Episode finished after 18 steps.`).


## Example
<!--- Please, update `examples/example.py` if you update example. -->

🆕Example code-snippet which uses [`configs/tasks/rearrange/pick.yaml`](configs/tasks/rearrange/pick.yaml) for configuration of task and agent.

```python
import habitat

# Load embodied AI task (RearrangePick) and a pre-specified virtual robot
env = habitat.Env(
    config=habitat.get_config("configs/tasks/rearrange/pick.yaml")
)

observations = env.reset()

# Step through environment with random actions
while not env.episode_over:
    observations = env.step(env.action_space.sample())

```

See [`examples/register_new_sensors_and_measures.py`](examples/register_new_sensors_and_measures.py) for an example of how to extend habitat-lab from _outside_ the source code.

## Documentation

Habitat Lab documentation is available [here](https://aihabitat.org/docs/habitat-lab/index.html).

For example, see [this page](https://aihabitat.org/docs/habitat-lab/quickstart.html) for a quickstart example.


## Docker Setup
We also provide a docker setup for habitat. This works on machines with an NVIDIA GPU and requires users to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). The following [Dockerfile](Dockerfile) was used to build the habitat docker. To setup the habitat stack using docker follow the below steps:

1. Pull the habitat docker image: `docker pull fairembodied/habitat:latest`

1. Start an interactive bash session inside the habitat docker: `docker run --runtime=nvidia -it fairhabitat/habitat:v1`

1. Activate the habitat conda environment: `source activate habitat`

1. Benchmark a forward only agent on the test scenes data: `cd habitat-api; python examples/benchmark.py`. This should print out an output like:
```bash
2019-02-25 02:39:48,680 initializing sim Sim-v0
2019-02-25 02:39:49,655 initializing task Nav-v0
spl: 0.000
```

## Details
An important objective of Habitat Lab is to make it easy for users to set up a variety of embodied agent tasks in 3D environments. The process of setting up a task involves using environment information provided by the simulator, connecting the information with a dataset (e.g. PointGoal targets, or question and answer pairs for Embodied QA) and providing observations which can be used by the agents. Keeping this primary objective in mind the core API defines the following key concepts as abstractions that can be extended:

* `Env`: the fundamental environment concept for Habitat. All the information needed for working on embodied tasks with a simulator is abstracted inside an Env. This class acts as a base for other derived environment classes. Env consists of three major components: a Simulator, a Dataset (containing Episodes), and a Task, and it serves to connects all these three components together.

* `Dataset`: contains a list of task-specific episodes from a particular data split and additional dataset-wide information. Handles loading and saving of a dataset to disk, getting a list of scenes, and getting a list of episodes for a particular scene.

* `Episode`: a class for episode specification that includes the initial position and orientation of an Agent, a scene id, a goal position and optionally shortest paths to the goal. An episode is a description of one task instance for the agent.

<p align="center">
  <img src='res/img/habitat_lab_structure.png' alt="teaser results" width="100%"/>
  <p align="center"><i>Architecture of Habitat Lab</i></p>
</p>

* `Task`: this class builds on top of the simulator and dataset. The criteria of episode termination and measures of success are provided by the Task.

* `Sensor`: a generalization of the physical Sensor concept provided by a Simulator, with the capability to provide Task-specific Observation data in a specified format.

* `Observation`: data representing an observation from a Sensor. This can correspond to physical sensors on an Agent (e.g. RGB, depth, semantic segmentation masks, collision sensors) or more abstract sensors such as the current agent state.

Note that the core functionality defines fundamental building blocks such as the API for interacting with the simulator backend, and receiving observations through Sensors. Concrete simulation backends, 3D datasets, and embodied agent baselines are implemented on top of the core API.

## Data
To make things easier we expect `data` folder of particular structure or symlink presented in habitat-lab working directory.

### Scenes datasets
| Scenes models | Extract path | Archive size |
| --- | --- | --- |
| 🆕[ReplicaCAD](#ReplicaCAD) | `data/scene_datasets/replica_cad/configs/scenes/{scene}.scene_instance.json` | 123 MB |
| 🆕[HM3D](#HM3D) | `data/scene_datasets/hm3d/{split}/00\d\d\d-{scene}/{scene}.basis.glb` | 130 GB |
| [Gibson](#Gibson) | `data/scene_datasets/gibson/{scene}.glb` | 1.5 GB |
| [MatterPort3D](#Matterport3D) | `data/scene_datasets/mp3d/{scene}/{scene}.glb` | 15 GB |

#### 🆕ReplicaCAD
Download [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/) dataset using download utility:
```
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset
```

#### 🆕Habitat Matterport
Download [HM3D](https://aihabitat.org/datasets/hm3d/) dataset using download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/089f6a41474f5470ca10222197c23693eef3a001/datasets/HM3D.md):
```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival
```

#### Matterport3D
The full Matterport3D (MP3D) dataset for use with Habitat can be downloaded using the official [Matterport3D](https://niessner.github.io/Matterport/) download script as follows: `python download_mp.py --task habitat -o data/scene_datasets/mp3d/`. You only need the habitat zip archive and not the entire Matterport3D dataset. Note that this download script requires python 2.7 to run. Extract the matterport data to `data/scene_datasets/mp3d`.

#### Gibson
Download the Habitat related Gibson dataset following the instructions [here](https://github.com/StanfordVL/GibsonEnv#database). After downloading extract the dataset to folder `habitat-lab/data/scene_datasets/gibson/` folder (this folder should contain the `.glb` files from Gibson).


### Task datasets
| Task | Scenes | Link | Extract path | Config to use | Archive size |
| --- | --- | --- | --- | --- | --- |
| 🆕[Rearrange Pick](https://arxiv.org/abs/2106.14405) | ReplicaCAD | [rearrange_pick_replica_cad_v0.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0.zip) | `data/datasets/rearrange_pick/replica_cad/v0/` |  [`datasets/rearrangpick/replica_cad.yaml`](configs/datasets/rearrangpick/replica_cad.yaml) | 11 MB |
| [Point goal navigation](https://arxiv.org/abs/1807.06757) | Gibson | [pointnav_gibson_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) | `data/datasets/pointnav/gibson/v1/` |  [`datasets/pointnav/gibson.yaml`](configs/datasets/pointnav/gibson.yaml) | 385 MB |
| 🆕[Point goal navigation](https://arxiv.org/abs/1807.06757) | Gibson 0+ (train) | [pointnav_gibson_0_plus_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_0_plus_v1.zip) | `data/datasets/pointnav/gibson/v1/` |  [`datasets/pointnav/gibson_0_plus.yaml`](configs/datasets/pointnav/gibson_0_plus.yaml) | 321 MB |
| [Point goal navigation corresponding to Sim2LoCoBot experiment configuration](https://arxiv.org/abs/1912.06321) | Gibson | [pointnav_gibson_v2.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip) | `data/datasets/pointnav/gibson/v2/` |  [`datasets/pointnav/gibson_v2.yaml`](configs/datasets/pointnav/gibson_v2.yaml) | 274 MB |
| [Point goal navigation](https://arxiv.org/abs/1807.06757) | MatterPort3D | [pointnav_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip) | `data/datasets/pointnav/mp3d/v1/` | [`datasets/pointnav/mp3d.yaml`](configs/datasets/pointnav/mp3d.yaml) | 400 MB |
| 🆕[Point goal navigation](https://arxiv.org/abs/1807.06757) | HM3D | [pointnav_hm3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/hm3d/v1/pointnav_hm3d_v1.zip) | `data/datasets/pointnav/hm3d/v1/` |  [`datasets/pointnav/hm3d.yaml`](configs/datasets/pointnav/hm3d.yaml) | 992 MB |
| [Object goal navigation](https://arxiv.org/abs/2006.13171) | MatterPort3D | [objectnav_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip) | `data/datasets/objectnav/mp3d/v1/` | [`datasets/objectnav/mp3d.yaml`](configs/datasets/objectnav/mp3d.yaml) | 170 MB |
| 🆕[Object goal navigation](https://arxiv.org/abs/2006.13171) | HM3D | [objectnav_hm3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip) | `data/datasets/objectnav/hm3d/v1/` |  [`datasets/objectnav/hm3d.yaml`](configs/datasets/objetnav/hm3d.yaml) | 154 MB |
| [Embodied Question Answering](https://embodiedqa.org/) | MatterPort3D | [eqa_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/eqa/mp3d/v1/eqa_mp3d_v1.zip) | `data/datasets/eqa/mp3d/v1/` | [`datasets/eqa/mp3d.yaml`](configs/datasets/eqa/mp3d.yaml) | 44 MB |
| [Visual Language Navigation](https://bringmeaspoon.org/) | MatterPort3D | [vln_r2r_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/vln/mp3d/r2r/v1/vln_r2r_mp3d_v1.zip) | `data/datasets/vln/mp3d/r2r/v1` | [`datasets/vln/mp3d_r2r.yaml`](configs/datasets/vln/mp3d_r2r.yaml) | 2.7 MB |
| [Image goal navigation](https://github.com/facebookresearch/habitat-lab/pull/333) | Gibson | [pointnav_gibson_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) | `data/datasets/pointnav/gibson/v1/` |  [`datasets/imagenav/gibson.yaml`](configs/datasets/imagenav/gibson.yaml) | 385 MB |
| [Image goal navigation](https://github.com/facebookresearch/habitat-lab/pull/333) | MatterPort3D | [pointnav_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip) | `data/datasets/pointnav/mp3d/v1/` | [`datasets/imagenav/mp3d.yaml`](configs/datasets/imagenav/mp3d.yaml) | 400 MB |

To use an episode dataset provide related config to the Env in [the example](#example) or use the config for [RL agent training](habitat_baselines/README.md#reinforcement-learning-rl).

## Baselines
Habitat Lab includes reinforcement learning (via PPO) and classical SLAM based baselines. For running PPO training on sample data and more details refer [habitat_baselines/README.md](habitat_baselines/README.md).

## Habitat-PyRobot
Habitat Lab supports deployment of models on a physical robot through PyRobot (https://github.com/facebookresearch/pyrobot). Please install the python3 version of PyRobot and refer to `habitat.sims.pyrobot.pyrobot` for instructions. This functionality allows deployment of models across simulation and reality.

## ROS-X-Habitat
ROS-X-Habitat (https://github.com/ericchen321/ros_x_habitat) is a framework that bridges the AI Habitat platform (Habitat Lab + Habitat Sim) with other robotics resources via ROS. Compared with Habitat-PyRobot, ROS-X-Habitat places emphasis on 1) leveraging Habitat Sim v2's physics-based simulation capability and 2) allowing roboticists to access simulation assets from ROS. The work has also been made public as a [paper](https://arxiv.org/abs/2109.07703).

Note that ROS-X-Habitat was developed, and is maintained by the Lab for Computational Intelligence at UBC; it has not yet been officially supported by the Habitat Lab team. Please refer to the framework's repository for docs and discussions.

## Acknowledgments
The Habitat project would not have been possible without the support and contributions of many individuals. We would like to thank Dmytro Mishkin, Xinlei Chen, Georgia Gkioxari, Daniel Gordon, Leonidas Guibas, Saurabh Gupta, Or Litany, Marcus Rohrbach, Amanpreet Singh, Devendra Singh Chaplot, Yuandong Tian, and Yuxin Wu for many helpful conversations and guidance on the design and development of the Habitat platform.

## License
Habitat Lab is MIT licensed. See the [LICENSE file](habitat_baselines/LICENSE) for details.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.
- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Gibson based task datasets, the code for generating such datasets, and trained models are distributed with [Gibson Terms of Use](https://storage.googleapis.com/gibson_material/Agreement%20GDS%2006-04-18.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

## References
1. 🆕[Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405) Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, Dhruv Batra. Advances in Neural Information Processing Systems (NeurIPS), 2021.
2. [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
