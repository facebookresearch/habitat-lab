# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     collapsed_sections: []
#     name: Habitat 2.0 Quick Start Tutorial
#     provenance: []
#   jupytext:
#     cell_metadata_filter: -all
#     formats: nb_python//py:percent,notebooks//ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.9.17
# ---

# %% [markdown]
# # Overview
# This tutorial covers the basics of using Habitat 2.0 including: setting up
# the environment, creating custom environments, and creating new episode
# datasets.

# %%
# Play a teaser video
from dataclasses import dataclass

from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    MeasurementConfig,
    ThirdRGBSensorConfig,
)

try:
    from IPython.display import IFrame

    # NOTE: this file is unreachable
    IFrame(
        src="https://drive.google.com/file/d/1ltrse38i8pnJPGAXlThylcdy8PMjUMKh/preview",
        width=640,
        height=480,
    )

except Exception:
    pass

# %%
# Imports
import os

import git
import gym
import imageio
import numpy as np
from hydra.core.config_store import ConfigStore

import habitat
import habitat.gym
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def insert_render_options(config):
    # Added settings to make rendering higher resolution for better visualization
    with habitat.config.read_write(config):
        config.habitat.simulator.concur_render = False
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {"third_rgb_sensor": ThirdRGBSensorConfig(height=512, width=512)}
        )
    return config


import importlib

# If the import block fails due to an error like "'PIL.TiffTags' has no attribute
# 'IFD'", then restart the Colab runtime instance and rerun this cell and the previous cell.
import PIL

importlib.reload(
    PIL.TiffTags  # type: ignore[attr-defined]
)  # To potentially avoid PIL problem

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(
    dir_path, "examples/tutorials/habitat_lab_visualization/"
)
os.makedirs(output_path, exist_ok=True)
os.chdir(dir_path)


# %% [markdown]
# # Local installation
# Follow the steps on the [Habitat Lab README](https://github.com/facebookresearch/habitat-lab#installation).

# %% [markdown]
# # Quickstart
#
# Start with a minimal environment interaction loop using the Habitat API. This sets up the environment, takes random episodes, and then saves a video once the episode ends.
#
# If this is your first time running Habitat 2.0 code, the datasets will automatically download which include the ReplicaCAD scenes, episode datasets, and object assets. To manually download this data, run `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`.

# %%
with habitat.Env(
    config=insert_render_options(
        habitat.get_config(
            os.path.join(
                dir_path,
                "habitat-lab/habitat/config/benchmark/rearrange/skills/pick.yaml",
            ),
        )
    )
) as env:
    observations = env.reset()  # noqa: F841

    print("Agent acting inside environment.")
    count_steps = 0
    # To save the video
    video_file_path = os.path.join(output_path, "example_interact.mp4")
    video_writer = imageio.get_writer(video_file_path, fps=30)

    while not env.episode_over:
        observations = env.step(env.action_space.sample())  # noqa: F841
        info = env.get_metrics()

        render_obs = observations_to_image(observations, info)
        render_obs = overlay_frame(render_obs, info)

        video_writer.append_data(render_obs)

        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))

    video_writer.close()
    if vut.is_notebook():
        vut.display_video(video_file_path)


# %% [markdown]
# ## Gym API
# You can also use environments through the Gym API. For more information about how to use the Gym API and the supported tasks, see [this tutorial](https://github.com/facebookresearch/habitat-lab/blob/main/examples/tutorials/colabs/habitat2_gym_tutorial.ipynb).

# %%
env = gym.make("HabitatRenderPick-v0")

video_file_path = os.path.join(output_path, "example_interact.mp4")
video_writer = imageio.get_writer(video_file_path, fps=30)

done = False
env.reset()
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    video_writer.append_data(env.render(mode="rgb_array"))

video_writer.close()
if vut.is_notebook():
    vut.display_video(video_file_path)


# %% [markdown]
# # Defining New Tasks
#
# We will define a task for the robot to navigate to and then pick up a target object in the environment. To support a new task we need:
# * A task of type `RearrangeTask` which implements the reset function.
# * Sensor definitions to populate the observation space.
# * Measurement definitions to define the reward, termination condition, and additional logging information.
#
# For other examples of task, sensor, and measurement definitions, [see here
# for existing tasks](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-lab/habitat/tasks/rearrange/sub_tasks). Tasks, sensors, and measurements are connected through a config file that defines the task.


# %%
@registry.register_task(name="RearrangeDemoNavPickTask-v0")
class NavPickTaskV1(RearrangeTask):
    """
    Primarily this is used to implement the episode reset functionality.
    Can also implement custom episode step functionality.
    """

    def reset(self, episode):
        self.target_object_index = np.random.randint(
            0, self._sim.get_n_targets()
        )
        start_pos = self._sim.pathfinder.get_random_navigable_point()
        self._sim.articulated_agent.base_pos = start_pos

        # Put any reset logic here.
        return super().reset(episode)


@registry.register_measure
class DistanceToTargetObject(Measure):
    """
    Gets the Euclidean distance to the target object from the end-effector.
    """

    cls_uuid: str = "distance_to_object"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DistanceToTargetObject.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, task, episode, **kwargs):
        ee_pos = self._sim.articulated_agent.ee_transform().translation

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()[idxs[task.target_object_index]]

        # Metric information is stored in the `self._metric` variable.
        self._metric = np.linalg.norm(scene_pos - ee_pos, ord=2, axis=-1)


@registry.register_measure
class NavPickReward(RearrangeReward):
    """
    For every new task, you NEED to implement a reward function.
    `RearrangeReward` automatically includes penalties for collisions into the reward function.
    """

    cls_uuid: str = "navpick_reward"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        # You can get you custom gonfiguration fields defined in NavPickRewardMeasurementConfig
        self._scaling_factor = config.scaling_factor
        super().__init__(sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavPickReward.cls_uuid

    def reset_metric(self, *args, task, episode, **kwargs):
        # Measurements can be computed from other measurements.
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DistanceToTargetObject.cls_uuid,
            ],
        )
        self.update_metric(*args, task=task, episode=episode, **kwargs)

    def update_metric(self, *args, task, episode, **kwargs):
        ee_to_object_distance = task.measurements.measures[
            DistanceToTargetObject.cls_uuid
        ].get_metric()

        self._metric = -ee_to_object_distance * self._scaling_factor


@registry.register_measure
class NavPickSuccess(Measure):
    """
    For every new task, you NEED to implement a "success" condition.
    """

    cls_uuid: str = "navpick_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavPickSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        # Check that the agent is holding the correct object.
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.target_object_index]
        self._metric = abs_targ_obj_idx == self._sim.grasp_mgr.snap_idx


@dataclass
class DistanceToTargetObjectMeasurementConfig(MeasurementConfig):
    type: str = "DistanceToTargetObject"


@dataclass
class NavPickRewardMeasurementConfig(MeasurementConfig):
    type: str = "NavPickReward"
    scaling_factor: float = 0.1
    # General Rearrange Reward config
    constraint_violate_pen: float = 10.0
    force_pen: float = 0.001
    max_force_pen: float = 1.0
    force_end_pen: float = 10.0
    count_coll_pen: float = -1.0
    max_count_colls: int = -1
    count_coll_end_pen: float = 1.0


@dataclass
class NavPickSuccessMeasurementConfig(MeasurementConfig):
    type: str = "NavPickSuccess"


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.distance_to_target_object",
    group="habitat/task/measurements",
    name="distance_to_target_object",
    node=DistanceToTargetObjectMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.nav_pick_reward",
    group="habitat/task/measurements",
    name="nav_pick_reward",
    node=NavPickRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.nav_pick_success",
    group="habitat/task/measurements",
    name="nav_pick_success",
    node=NavPickSuccessMeasurementConfig,
)


# %% [markdown]
# We now add all the previously defined task, sensor, and measurement
# definitions to a config file to finish defining the new Habitat task. For
# examples of more configs [see here](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-lab/habitat/config/habitat/task/rearrange).
#
# This config also defines the action space through the `task.actions` key. You
# can substitute different base control actions from
# [here](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/rearrange/actions/actions.py),
# different arm control actions [from
# here](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/rearrange/actions/actions.py),
# and different grip actions [from here](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/rearrange/actions/grip_actions.py).

# %%
cfg_txt = """
# @package _global_

defaults:
  - /habitat: habitat_config_base
  - /habitat/simulator/agents@habitat.simulator.agents.main_agent: agent_base
  - /habitat/simulator/sim_sensors@habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor: head_rgb_sensor
  - /habitat/task: task_config_base
  - /habitat/task/actions:
    - arm_action
    - base_velocity
  - /habitat/task/measurements:
    - articulated_agent_force
    - force_terminate
    - distance_to_target_object
    - nav_pick_reward
    - nav_pick_success
  - /habitat/task/lab_sensors:
    - target_start_sensor
    - joint_sensor
  - /habitat/dataset/rearrangement: replica_cad

habitat:
  environment:
    # Number of steps within an episode.
    max_episode_steps: 200
  task:
    type: RearrangeDemoNavPickTask-v0
    # Measurements
    measurements:
      distance_to_target_object:
        type: "DistanceToTargetObject"
      articulated_agent_force:
        type: "RobotForce"
        min_force: 20.0
      force_terminate:
        type: "ForceTerminate"
        # Maximum amount of allowed force in Newtons.
        max_accum_force: 5000.0
      nav_pick_reward:
        type: "NavPickReward"
        scaling_factor: 0.1
        # General Rearrange Reward config
        constraint_violate_pen: 10.0
        force_pen: 0.001
        max_force_pen: 1.0
        force_end_pen: 10.0
      nav_pick_success:
        type: "NavPickSuccess"
    actions:
      # Define the action space.
      arm_action:
        type: "ArmAction"
        arm_controller: "ArmRelPosAction"
        grip_controller: "MagicGraspAction"
        arm_joint_dimensionality: 7
        grasp_thresh_dist: 0.15
        disable_grip: False
        delta_pos_limit: 0.0125
        ee_ctrl_lim: 0.015
      base_velocity:
        type: "BaseVelAction"
        lin_speed: 12.0
        ang_speed: 12.0
        allow_dyn_slide: True
        allow_back: True
  simulator:
    type: RearrangeSim-v0
    additional_object_paths:
      - "data/objects/ycb/configs/"
    debug_render: False
    concur_render: False
    auto_sleep: False
    agents:
      main_agent:
        height: 1.5
        is_set_start_state: False
        radius: 0.1
        sim_sensors:
          head_rgb_sensor:
            height: 128
            width: 128
        start_position: [0, 0, 0]
        start_rotation: [0, 0, 0, 1]
        articulated_agent_urdf: ./data/robots/hab_fetch/robots/hab_fetch.urdf
        articulated_agent_type: "FetchRobot"

    # Agent setup
    # ARM_REST: [0.6, 0.0, 0.9]
    ctrl_freq: 120.0
    ac_freq_ratio: 4

    # Grasping
    hold_thresh: 0.09
    grasp_impulse: 1000.0

    habitat_sim_v0:
      allow_sliding: True
      enable_physics: True
      gpu_device_id: 0
      gpu_gpu: False
      physics_config_file: ./data/default.physics_config.json
  dataset:
    type: RearrangeDataset-v0
    split: train
    # The dataset to use. Later we will generate our own dataset.
    data_path: data/datasets/replica_cad/rearrange/v2/{split}/all_receptacles_10k_1k.json.gz
    scenes_dir: "data/replica_cad/"
"""
nav_pick_cfg_path = os.path.join(data_path, "nav_pick_demo.yaml")
with open(nav_pick_cfg_path, "w") as f:
    f.write(cfg_txt)

# %% [markdown]
# The new task can then be imported via the yaml file.

# %%
with habitat.Env(
    config=insert_render_options(habitat.get_config(nav_pick_cfg_path))
) as env:
    env.reset()

    print("Agent acting inside environment.")
    count_steps = 0
    # To save the video
    video_file_path = os.path.join(output_path, "example_interact.mp4")
    video_writer = imageio.get_writer(video_file_path, fps=30)

    while not env.episode_over:
        action = env.action_space.sample()
        observations = env.step(action)  # noqa: F841
        info = env.get_metrics()

        render_obs = observations_to_image(observations, info)
        render_obs = overlay_frame(render_obs, info)

        video_writer.append_data(render_obs)

        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))

    video_writer.close()
    if vut.is_notebook():
        vut.display_video(video_file_path)

# %% [markdown]
# # Dataset Generation
# The previously defined task uses an included default `all_receptacles_10k_1k.json.gz` dataset which places objects on any receptacle. The episode `.json.gz` dataset defines where
# objects are placed and their rearrangement target positions. New episode
# datasets are generated with the [run_episode_generator.py](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/datasets/rearrange/run_episode_generator.py) script. In this example, we will define a new episode dataset where a single object spawns on the table with its goal also on the table.

# %%
dataset_cfg_txt = """
---
dataset_path: "data/replica_cad/replicaCAD.scene_dataset_config.json"
additional_object_paths:
  - "data/objects/ycb/configs/"
scene_sets:
  -
    name: "v3_sc"
    included_substrings:
      - "v3_sc"
    excluded_substrings: []
    comment: "This set (v3_sc) selects all 105 ReplicaCAD variations with static furniture."

object_sets:
  -
    name: "kitchen"
    included_substrings:
      - "002_master_chef_can"
      - "003_cracker_box"
    excluded_substrings: []
    comment: "Leave included_substrings empty to select all objects."

receptacle_sets:
  -
    name: "table"
    included_object_substrings:
      - "frl_apartment_table_01"
    excluded_object_substrings: []
    included_receptacle_substrings:
      - ""
    excluded_receptacle_substrings: []
    comment: "The empty substrings act like wildcards, selecting all receptacles for all objects."

scene_sampler:
  type: "subset"
  params:
    scene_sets: ["v3_sc"]
  comment: "Samples from ReplicaCAD 105 variations with static furniture."


object_samplers:
  -
    name: "kitchen_counter"
    type: "uniform"
    params:
      object_sets: ["kitchen"]
      receptacle_sets: ["table"]
      num_samples: [1, 1]
      orientation_sampling: "up"

object_target_samplers:
  -
    name: "kitchen_counter_targets"
    type: "uniform"
    params:
      object_samplers: ["kitchen_counter"]
      receptacle_sets: ["table"]
      num_samples: [1, 1]
      orientation_sampling: "up"
"""
nav_pick_cfg_path = os.path.join(data_path, "nav_pick_dataset.yaml")
with open(nav_pick_cfg_path, "w") as f:
    f.write(dataset_cfg_txt)

# %%
# !python -m habitat.datasets.rearrange.run_episode_generator --run --config {nav_pick_cfg_path} --num-episodes 10 --out data/nav_pick.json.gz

# %% [markdown]
# To use this dataset set `dataset.data_path = data/nav_pick.json.gz` in the task config. See the full set of possible objects, receptacles, and scenes with `python -m habitat.datasets.rearrange.run_episode_generator --list`
