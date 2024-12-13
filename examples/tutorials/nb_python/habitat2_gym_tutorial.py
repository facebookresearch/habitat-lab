# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     collapsed_sections: []
#     name: Habitat 2.0 Gym Tutorial
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
# # Habitat 2.0 Gym API
# This tutorial covers how to use Habitat 2.0 environments as standard gym environments.
# See [here for Habitat 2.0 installation instructions and more tutorials.](https://aihabitat.org/docs/habitat2/)

# %%
import os

import git

if "COLAB_GPU" in os.environ:
    print("Setting Habitat base path")
    # %env HABLAB_BASE_CFG_PATH=/content/habitat-lab
    import importlib

    import PIL

    importlib.reload(PIL.TiffTags)  # type: ignore[attr-defined]

import imageio

# Video rendering utility.
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
output_path = os.path.join(
    dir_path, "examples/tutorials/habitat_lab_visualization/"
)
os.makedirs(output_path, exist_ok=True)
os.chdir(dir_path)
# If the import block below fails due to an error like "'PIL.TiffTags' has no attribute
# 'IFD'", then restart the Colab runtime instance and rerun this cell and the previous cell.

# %%
# The ONLY two lines you need to add to start importing Habitat 2.0 Gym environments.
import gym

# flake8: noqa
import habitat.gym

# %% [markdown]
# # Simple Example
# This example sets up the Pick task in render mode which includes a high resolution camera in the scene for visualization.

# %%
env = gym.make("HabitatRenderPick-v0")

video_file_path = os.path.join(output_path, "example_interact.mp4")
video_writer = imageio.get_writer(video_file_path, fps=30)

done = False
env.reset()
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    video_writer.append_data(env.render("rgb_array"))

video_writer.close()
if vut.is_notebook():
    vut.display_video(video_file_path)

env.close()

# %% [markdown]
# # Environment Options
# To create the environment in performance mode remove `Render` from the environment ID string. The environment ID follows the format: `Habitat[Render?][Task Name]-v0`. All the supported environment IDs are listed below. The `Render` option can always be added to include the higher resolution 3rd POV camera for visualization.
#
# * Skills:
#     * `HabitatPick-v0`
#     * `HabitatPlace-v0`
#     * `HabitatCloseCab-v0`
#     * `HabitatCloseFridge-v0`
#     * `HabitatOpenCab-v0`
#     * `HabitatOpenFridge-v0`
#     * `HabitatNavToObj-v0`
#     * `HabitatReachState-v0`
# * Home Assistant Benchmark (HAB) tasks:
#     * `HabitatTidyHouse-v0`
#     * `HabitatPrepareGroceries-v0`
#     * `HabitatSetTable-v0`
#
# The Gym environments are automatically registered from the RL training configurations under ["habitat-lab/habitat/config/benchmark/rearrange"](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-lab/habitat/config/benchmark/rearrange). The observation keys in `habitat.gym.obs_keys` are what is returned in the observation space.
#
# An example of these different observation spaces is demonstrated below:

# %%
# Dictionary observation space
env = gym.make("HabitatPick-v0")
print(
    "Pick observation space",
    {k: v.shape for k, v in env.observation_space.spaces.items()},
)
env.close()

# Array observation space
env = gym.make("HabitatReachState-v0")
print("Reach observation space", env.observation_space)
env.close()

# %% [markdown]
# # Environment Configuration
#
# You can also modify the config specified in the YAML file through `gym.make` by passing the `override_options` argument. Here is an example of changing the gripper type to use the suction grasp in the Pick Task.

# %%
env = gym.make(
    "HabitatPick-v0",
    override_options=[
        "habitat.task.actions.arm_action.grip_controller=SuctionGraspAction",
    ],
)
print("Action space with suction grip", env.action_space)
env.close()
