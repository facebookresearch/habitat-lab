# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     collapsed_sections: []
#     name: habitat-lab.ipynb
#     provenance: []
#   jupytext:
#     cell_metadata_filter: -all
#     formats: nb_python//py:percent,colabs//ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
# @title Installation

# !curl -L https://raw.githubusercontent.com/facebookresearch/habitat-sim/master/examples/colab_utils/colab_install.sh | NIGHTLY=true bash -s
# !wget -c http://dl.fbaipublicfiles.com/habitat/mp3d_example.zip && unzip -o mp3d_example.zip -d /content/habitat-sim/data/scene_datasets/mp3d/

# %%
# !pip uninstall --yes pyopenssl
# !pip install pyopenssl

# %%
# @title Colab Setup and Imports { display-mode: "form" }
# @markdown (double click to see the code)

import os, sys

try:
    import habitat_sim
except ImportError:
    import sys, os

    if "google.colab" in sys.modules:
        # ADDS conda installed libraries to the PYTHONPATH
        conda_path = "/usr/local/lib/python3.6/site-packages/"
        user_path = "/root/.local/lib/python3.6/site-packages/"
        sys.path.insert(0, conda_path)
        sys.path.insert(0, user_path)
os.chdir("/content/habitat-sim")

import habitat_sim.utils.common as utils
from habitat_sim.utils import viz_utils as vut
import magnum as mn
import math

import random

# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
from gym import spaces

if "google.colab" in sys.modules:
    # This tells imageio to use the system FFMPEG that has hardware acceleration.
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

sys.path.append("/content/habitat-api/")

import habitat
from habitat.core.embodied_task import EmbodiedTask
from habitat.tasks.nav.nav import NavigationTask
from habitat.core.registry import registry
from habitat.core.logging import logger

import habitat_baselines
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat_baselines.common.baseline_registry import baseline_registry

# %%
# @title Define Observation Display Utility Function { display-mode: "form" }

# @markdown A convenient function that displays sensor observations with matplotlib.

# @markdown (double click to see the code)

from PIL import Image

# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


# %% [markdown]
# ## Setup PointNav Task

# %%
# cat "/content/habitat-api/configs/test/habitat_all_sensors_test.yaml"

# %%
config = habitat.get_config(
    config_paths="/content/habitat-api/configs/test/habitat_all_sensors_test.yaml"
)

env = habitat.Env(config=config)

# %%
action = None
obs = env.reset()
valid_actions = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "STOP"]

while action != "STOP":
    display_sample(obs["rgb"])
    print(
        "distance to goal: {:.2f}".format(obs["pointgoal_with_gps_compass"][0])
    )
    print(
        "angle to goal (radians): {:.2f}".format(
            obs["pointgoal_with_gps_compass"][1]
        )
    )
    action = input(
        "enter action out of {}:\n".format(", ".join(valid_actions))
    )
    assert action in valid_actions, (
        "invalid action {} entered, choose one amongst "
        + ",".join(valid_actions)
    )
    obs = env.step({"action": action,})

env.close()

# %%
print(env.get_metrics())

# %% [markdown]
# ## RL Training

# %%
config = get_baselines_config(
    "/content/habitat-api/habitat_baselines/config/pointnav/ppo_pointnav_example.yaml"
)

# %%
# set random seeds

seed = "42"  # @param {type:"string"}
num_updates = "20"  # @param {type:"string"}

config.defrost()
config.TASK_CONFIG.SEED = int(seed)
config.NUM_UPDATES = int(num_updates)
config.LOG_INTERVAL = 1
config.freeze()

random.seed(config.TASK_CONFIG.SEED)
np.random.seed(config.TASK_CONFIG.SEED)

# %%
trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
trainer = trainer_init(config)
trainer.train()

# %%
# @markdown (double click to see the code)

# example tensorboard visualization
# for more details refer to [link](https://github.com/facebookresearch/habitat-api/tree/master/habitat_baselines#additional-utilities).

from IPython import display

with open(
    "/content/habitat-api/res/img/tensorboard_video_demo.gif", "rb"
) as f:
    display.display(display.Image(data=f.read(), format="png"))

# %% [markdown]
# ## Key Concepts
#
# All the concepts link to their definitions:
#
# 1. [`habitat.sims.habitat_simulator.HabitatSim`](https://github.com/facebookresearch/habitat-api/blob/master/habitat/sims/habitat_simulator/habitat_simulator.py#L159)
# Thin wrapper over `habitat_sim` providing seamless integration with experimentation framework.
#
#
# 2. [`habitat.core.env.Env`](https://github.com/facebookresearch/habitat-api/blob/master/habitat/core/env.py)
# Abstraction for the universe of agent, task and simulator. Agents that you train and evaluate operate inside the environment.
#
#
# 3. [`habitat.core.env.RLEnv`](https://github.com/facebookresearch/habitat-api/blob/71d409ab214a7814a9bd9b7e44fd25f57a0443ba/habitat/core/env.py#L278)
# Extends the `Env` class for reinforcement learning by defining the reward and other required components.
#
#
# 4. [`habitat.core.embodied_task.EmbodiedTask`](https://github.com/facebookresearch/habitat-api/blob/71d409ab214a7814a9bd9b7e44fd25f57a0443ba/habitat/core/embodied_task.py#L242)
# Defines the task that the agent needs to solve. This class holds the definition of observation space, action space, measures, simulator usage. Eg: PointNav, ObjectNav.
#
#
# 5. [`habitat.core.dataset.Dataset`](https://github.com/facebookresearch/habitat-api/blob/4b6da1c4f8eb287cea43e70c50fe1d615a261198/habitat/core/dataset.py#L63)
# Wrapper over information required for the dataset of embodied task, contains definition and interaction with an `episode`.
#
#
# 6. [`habitat.core.embodied_task.Measure`](https://github.com/facebookresearch/habitat-api/blob/master/habitat/core/embodied_task.py#L82)
# Defines the metrics for embodied task, eg: [SPL](https://github.com/facebookresearch/habitat-api/blob/d0db1b55be57abbacc5563dca2ca14654c545552/habitat/tasks/nav/nav.py#L533).
#
#
# 7. [`habitat_baselines`](https://github.com/facebookresearch/habitat-api/tree/71d409ab214a7814a9bd9b7e44fd25f57a0443ba/habitat_baselines)
# RL, SLAM, heuristic baseline implementations for the different embodied tasks.

# %% [markdown]
# ## Create a new Task

# %%
config = habitat.get_config(
    config_paths="/content/habitat-api/configs/test/habitat_all_sensors_test.yaml"
)


@registry.register_task(name="TestNav-v0")
class NewNavigationTask(NavigationTask):
    def __init__(self, config, sim, dataset):
        logger.info("Creating a new type of task")
        super().__init__(config=config, sim=sim, dataset=dataset)

    def _check_episode_is_active(self, *args, **kwargs):
        logger.info(
            "Current agent position: {}".format(self._sim.get_agent_state())
        )
        collision = self._sim.previous_step_collided
        stop_called = not getattr(self, "is_stop_called", False)
        return collision or stop_called


config.defrost()
config.TASK.TYPE = "TestNav-v0"
config.freeze()

env = habitat.Env(config=config)

# %%
action = None
env.reset()
valid_actions = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "STOP"]

while env.episode_over is not True:
    display_sample(obs["rgb"])
    action = input(
        "enter action out of {}:\n".format(", ".join(valid_actions))
    )
    assert action in valid_actions, (
        "invalid action {} entered, choose one amongst "
        + ",".join(valid_actions)
    )
    obs = env.step({"action": action, "action_args": None,})
    print("Episode over:", env.episode_over)

env.close()


# %% [markdown]
# ## Create a new Sensor

# %%
@registry.register_sensor(name="agent_position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args, **kwargs):
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args, **kwargs):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations, *args, episode, **kwargs):
        return self._sim.get_agent_state().position


# %%
config = habitat.get_config(
    config_paths="/content/habitat-api/configs/test/habitat_all_sensors_test.yaml"
)

config.defrost()
# Now define the config for the sensor
config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
# Use the custom name
config.TASK.AGENT_POSITION_SENSOR.TYPE = "agent_position_sensor"
# Add the sensor to the list of sensors in use
config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
config.freeze()

env = habitat.Env(config=config)

# %%
obs = env.reset()

# %%
obs.keys()

# %%
print(obs["agent_position"])

# %%
env.close()

# %% [markdown]
# ## Create a new Agent

# %%
# An example agent which can be submitted to habitat-challenge.
# To participate and for more details refer to:
# - https://aihabitat.org/challenge/2020/
# - https://github.com/facebookresearch/habitat-challenge


class ForwardOnlyAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            action = HabitatSimActions.MOVE_FORWARD
        return {"action": action}


# %% [markdown]
# ### Other Examples
#
# [Create a new action space](https://github.com/facebookresearch/habitat-api/blob/master/examples/new_actions.py)

# %%
# @title Sim2Real with Habitat { display-mode: "form" }

from IPython.display import HTML

HTML(
    '<iframe width="560" height="315" src="https://www.youtube.com/embed/Hun2rhgnWLU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
)

# %% [markdown]
# Deploy habitat-sim trained models on real robots with the [habitat-pyrobot bridge](https://github.com/facebookresearch/habitat-api/blob/71d409ab214a7814a9bd9b7e44fd25f57a0443ba/habitat/sims/pyrobot/pyrobot.py)
#
# ```python
# # Are we in sim or reality?
# if args.use_robot: # Use LoCoBot via PyRobot
#     config.SIMULATOR.TYPE = "PyRobot-Locobot-v0"
# else: # Use simulation
#     config.SIMULATOR.TYPE = "Habitat-Sim-v0"
# ```
#
# Paper: [https://arxiv.org/abs/1912.06321](https://arxiv.org/abs/1912.06321)
