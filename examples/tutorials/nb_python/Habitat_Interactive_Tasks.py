# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     collapsed_sections: []
#     name: Habitat Interactive Tasks
#     provenance: []
#     toc_visible: true
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
#     version: 3.7.3
# ---

# %% [markdown]
# # Furniture Rearrangement - How to setup a new interaction task in Habitat-Lab
#
# This tutorial demonstrates how to setup a new task in Habitat that utilizes interaction capabilities in Habitat Simulator.
#
# ![teaser](https://drive.google.com/uc?id=1pupGvb4dGefd0T_23GpeDkkcIocDHSL_)
#
# ## Task Definition:
# The working example in this demo will be the task of **Furniture Rearrangement** - The agent will be randomly spawned in an environment in which the furniture are initially displaced from their desired position. The agent is tasked with navigating the environment, picking furniture and putting them in the desired position. To keep the tutorial simple and easy to follow, we will rearrange just a single object.
#
# To setup this task, we will build on top of existing API in Habitat-Simulator and Habitat-Lab. Here is a summary of all the steps involved in setting up this task:
#
# 1. **Setup the Simulator**: Using existing functionalities of the Habitat-Sim, we can add or remove objects from the scene. We will use these methods to spawn the agent and the objects at some pre-defined initial configuration.
# 2. **Create a New Dataset**: We will define a new dataset class to save / load a list of episodes for the agent to train and evaluate on.
# 3. **Grab / Release Action**: We will add the "grab/release" action to the agent's action space to allow the agent to pickup / drop an object under a crosshair.
# 4. **Extend the Simulator Class**: We will extend the Simulator Class to add support for new actions implemented in previous step and add other additional utility functions
# 5. **Create a New Task**: Create a new task definition, implement new *sensors* and *metrics*.
# 6. **Train an RL agent**: We will define rewards for this task and utilize it to train an RL agent using the PPO algorithm.
#
# Let's get started!

# %%
# @title Installation { display-mode: "form" }
# @markdown (double click to show code).

# !curl -L https://raw.githubusercontent.com/facebookresearch/habitat-sim/master/examples/colab_utils/colab_install.sh | NIGHTLY=true bash -s
# %cd /content

# !gdown --id 1Pc-J6pZzXEd8RSeLM94t3iwO8q_RQ853
# !unzip -o /content/coda.zip -d /content/habitat-sim/data/scene_datasets

# reload the cffi version
import sys

if "google.colab" in sys.modules:
    import importlib

    import cffi

    importlib.reload(cffi)

# %%
# @title Path Setup and Imports { display-mode: "form" }
# @markdown (double click to show code).

# %cd /content/habitat-lab

## [setup]
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional, Type

import attr
import cv2
import git
import magnum as mn
import numpy as np

# %matplotlib inline
from matplotlib import pyplot as plt
from PIL import Image

import habitat
import habitat_sim
from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut

if "google.colab" in sys.modules:
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
# %cd $dir_path
data_path = os.path.join(dir_path, "data")
output_directory = "data/tutorials/output/"  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument(
        "--no-make-video", dest="make_video", action="store_false"
    )
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video
else:
    show_video = False
    make_video = False
    display = False

if make_video and not os.path.exists(output_path):
    os.makedirs(output_path)


# %%
# @title Util functions to visualize observations
# @markdown - `make_video_cv2`: Renders a video from a list of observations
# @markdown - `simulate`: Runs simulation for a given amount of time at 60Hz
# @markdown - `simulate_and_make_vid` Runs simulation and creates video


def make_video_cv2(
    observations, cross_hair=None, prefix="", open_vid=True, fps=60
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(videodims)
    video_file = output_path + prefix + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = vut.get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        rgb_im_1st_person = cv2.cvtColor(ob["rgb"], cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        vut.display_video(video_file)


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations


# convenience wrapper for simulate and make_video_cv2
def simulate_and_make_vid(sim, crosshair, prefix, dt=1.0, open_vid=True):
    observations = simulate(sim, dt)
    make_video_cv2(observations, crosshair, prefix=prefix, open_vid=open_vid)


def display_sample(
    rgb_obs,
    semantic_obs=np.array([]),
    depth_obs=np.array([]),
    key_points=None,  # noqa: B006
):
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
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(
                    point[0], point[1], marker="o", markersize=10, alpha=0.8
                )
        plt.imshow(data)

    plt.show(block=False)


# %% [markdown]
# ## 1. Setup the Simulator
#
# ---
#
#

# %%
# @title Setup simulator configuration
# @markdown We'll start with setting up simulator with the following configurations
# @markdown - The simulator will render both RGB, Depth observations of 256x256 resolution.
# @markdown - The actions available will be `move_forward`, `turn_left`, `turn_right`.


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.default_agent_id = settings["default_agent_id"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.physics_config_file = settings["physics_config_file"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgb_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


settings = {
    "max_frames": 10,
    "width": 256,
    "height": 256,
    "scene": "data/scene_datasets/coda/coda.glb",
    "default_agent_id": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "rgb": True,  # RGB sensor
    "depth": True,  # Depth sensor
    "seed": 1,
    "enable_physics": True,
    "physics_config_file": "data/default.physics_config.json",
    "silent": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "save_png": True,
}

cfg = make_cfg(settings)


# %%
# @title Spawn the agent at a pre-defined location


def init_agent(sim):
    agent_pos = np.array([-0.15776923, 0.18244143, 0.2988735])

    # Place the agent
    sim.agents[0].scene_node.translation = agent_pos
    agent_orientation_y = -40
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0)
    )


cfg.sim_cfg.default_agent_id = 0
with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    if make_video:
        # Visualize the agent's initial position
        simulate_and_make_vid(
            sim, None, "sim-init", dt=1.0, open_vid=show_video
        )


# %%
# @title Set the object's initial and final position
# @markdown Defines two utility functions:
# @markdown - `remove_all_objects`: This will remove all objects from the scene
# @markdown - `set_object_in_front_of_agent`: This will add an object in the scene in front of the agent at the specified distance.

# @markdown Here we add a chair *3.0m* away from the agent and the task is to place the agent at the desired final position which is *7.0m* in front of the agent.


def remove_all_objects(sim):
    for obj_id in sim.get_existing_object_ids():
        sim.remove_object(obj_id)


def set_object_in_front_of_agent(sim, obj_id, z_offset=-1.5):
    r"""
    Adds an object in front of the agent at some distance.
    """
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    obj_translation = agent_transform.transform_point(
        np.array([0, 0, z_offset])
    )
    sim.set_translation(obj_translation, obj_id)

    obj_node = sim.get_object_scene_node(obj_id)
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )

    # also account for collision margin of the scene
    scene_collision_margin = 0.04
    y_translation = mn.Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    )
    sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)


def init_objects(sim):
    # Manager of Object Attributes Templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects"))
    )

    # Add a chair into the scene.
    obj_path = "test_assets/objects/chair"
    chair_template_id = obj_attr_mgr.load_object_configs(
        str(os.path.join(data_path, obj_path))
    )[0]
    chair_attr = obj_attr_mgr.get_template_by_ID(chair_template_id)
    obj_attr_mgr.register_template(chair_attr)

    # Object's initial position 3m away from the agent.
    object_id = sim.add_object_by_handle(chair_attr.handle)
    set_object_in_front_of_agent(sim, object_id, -3.0)
    sim.set_object_motion_type(
        habitat_sim.physics.MotionType.STATIC, object_id
    )

    # Object's final position 7m away from the agent
    goal_id = sim.add_object_by_handle(chair_attr.handle)
    set_object_in_front_of_agent(sim, goal_id, -7.0)
    sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, goal_id)

    return object_id, goal_id


with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    init_objects(sim)

    # Visualize the scene after the chair is added into the scene.
    if make_video:
        simulate_and_make_vid(
            sim, None, "object-init", dt=1.0, open_vid=show_video
        )


# %% [markdown]
# ## Rearrangement Dataset
# ![dataset](https://drive.google.com/uc?id=1y0qS0MifmJsZ0F4jsRZGI9BrXzslFLn7)
#
# In the previous section, we created a single episode of the rearrangement task. Let's define a format to store all the necessary information about a single episode. It should store the *scene* the episode belongs to, *initial spawn position and orientation* of the agent, *object type*, object's *initial position and orientation* as well as *final position and orientation*.
#
# The format will be as follows:
# ```
# {
#   'episode_id': 0,
#   'scene_id': 'data/scene_datasets/coda/coda.glb',
#   'goals': {
#     'position': [4.34, 0.67, -5.06],
#     'rotation': [0.0, 0.0, 0.0, 1.0]
#    },
#   'objects': {
#     'object_id': 0,
#     'object_template': 'data/test_assets/objects/chair',
#     'position': [1.77, 0.67, -1.99],
#     'rotation': [0.0, 0.0, 0.0, 1.0]
#   },
#   'start_position': [-0.15, 0.18, 0.29],
#   'start_rotation': [-0.0, -0.34, -0.0, 0.93]}
# }
# ```
# Once an episode is defined, a dataset will just be a collection of such episodes. For simplicity, in this notebook, the dataset will only contain one episode defined above.
#

# %%
# @title Create a new dataset
# @markdown Utility functions to define and save the dataset for the rearrangement task


def get_rotation(sim, object_id):
    quat = sim.get_rotation(object_id)
    return np.array(quat.vector).tolist() + [quat.scalar]


def init_episode_dict(episode_id, scene_id, agent_pos, agent_rot):
    episode_dict = {
        "episode_id": episode_id,
        "scene_id": "data/scene_datasets/coda/coda.glb",
        "start_position": agent_pos,
        "start_rotation": agent_rot,
        "info": {},
    }
    return episode_dict


def add_object_details(sim, episode_dict, obj_id, object_template, object_id):
    object_template = {
        "object_id": obj_id,
        "object_template": object_template,
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["objects"] = object_template
    return episode_dict


def add_goal_details(sim, episode_dict, object_id):
    goal_template = {
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["goals"] = goal_template
    return episode_dict


# set the number of objects to 1 always for now.
def build_episode(sim, episode_num, object_id, goal_id):
    episodes = {"episodes": []}
    for episode in range(episode_num):
        agent_state = sim.get_agent(0).get_state()
        agent_pos = np.array(agent_state.position).tolist()
        agent_quat = agent_state.rotation
        agent_rot = np.array(agent_quat.vec).tolist() + [agent_quat.real]
        episode_dict = init_episode_dict(
            episode, settings["scene"], agent_pos, agent_rot
        )

        object_attr = sim.get_object_initialization_template(object_id)
        object_path = os.path.relpath(
            os.path.splitext(object_attr.render_asset_handle)[0]
        )

        episode_dict = add_object_details(
            sim, episode_dict, 0, object_path, object_id
        )
        episode_dict = add_goal_details(sim, episode_dict, goal_id)
        episodes["episodes"].append(episode_dict)

    return episodes


with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    object_id, goal_id = init_objects(sim)

    episodes = build_episode(sim, 1, object_id, goal_id)

    dataset_content_path = "data/datasets/rearrangement/coda/v1/train/"
    if not os.path.exists(dataset_content_path):
        os.makedirs(dataset_content_path)

    with gzip.open(
        os.path.join(dataset_content_path, "train.json.gz"), "wt"
    ) as f:
        json.dump(episodes, f)

    print(
        "Dataset written to {}".format(
            os.path.join(dataset_content_path, "train.json.gz")
        )
    )


# %%
# @title Dataset class to read the saved dataset in Habitat-Lab.
# @markdown To read the saved episodes in Habitat-Lab, we will extend the `Dataset` class and the `Episode` base class. It will help provide all the relevant details about the episode through a consistent API to all downstream tasks.

# @markdown - We will first create a `RearrangementEpisode` by extending the `NavigationEpisode` to include additional information about object's initial configuration and desired final configuration.
# @markdown - We will then define a `RearrangementDatasetV0` class that builds on top of `PointNavDatasetV1` class to read the JSON file stored earlier and initialize a list of `RearrangementEpisode`.

from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav import NavigationEpisode


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementObjectSpec(RearrangementSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """
    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_template: Optional[str] = attr.ib(
        default="data/test_assets/objects/chair"
    )


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation
    of agent, all goal specifications, all object specifications

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goal: object's goal position and rotation
        object: object's start specification defined with object type,
            position, and rotation.
    """
    objects: RearrangementObjectSpec = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: RearrangementSpec = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_dataset(name="RearrangementDataset-v0")
class RearrangementDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[RearrangementEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = RearrangementEpisode(**episode)
            rearrangement_episode.episode_id = str(i)

            if scenes_dir is not None:
                if rearrangement_episode.scene_id.startswith(
                    DEFAULT_SCENE_PATH_PREFIX
                ):
                    rearrangement_episode.scene_id = (
                        rearrangement_episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]
                    )

                rearrangement_episode.scene_id = os.path.join(
                    scenes_dir, rearrangement_episode.scene_id
                )

            rearrangement_episode.objects = RearrangementObjectSpec(
                **rearrangement_episode.objects
            )
            rearrangement_episode.goals = RearrangementSpec(
                **rearrangement_episode.goals
            )

            self.episodes.append(rearrangement_episode)


# %%
# @title Load the saved dataset using the Dataset class
config = habitat.get_config("configs/datasets/pointnav/habitat_test.yaml")
config.defrost()
config.DATASET.DATA_PATH = (
    "data/datasets/rearrangement/coda/v1/{split}/{split}.json.gz"
)
config.DATASET.TYPE = "RearrangementDataset-v0"
config.freeze()

dataset = RearrangementDatasetV0(config.DATASET)

# check if the dataset got correctly deserialized
assert len(dataset.episodes) == 1

assert dataset.episodes[0].objects.position == [
    1.770593523979187,
    0.6726829409599304,
    -1.9992598295211792,
]
assert dataset.episodes[0].objects.rotation == [0.0, 0.0, 0.0, 1.0]
assert (
    dataset.episodes[0].objects.object_template
    == "data/test_assets/objects/chair"
)

assert dataset.episodes[0].goals.position == [
    4.3417439460754395,
    0.6726829409599304,
    -5.0634379386901855,
]
assert dataset.episodes[0].goals.rotation == [0.0, 0.0, 0.0, 1.0]


# %% [markdown]
# ## Implement Grab/Release Action

# %%
# @title RayCast utility to implement Grab/Release Under Cross-Hair Action
# @markdown Cast a ray in the direction of crosshair from the camera and check if it collides with another object within a certain distance threshold


def raycast(sim, sensor_name, crosshair_pos=(128, 128), max_distance=2.0):
    r"""Cast a ray in the direction of crosshair and check if it collides
    with another object within a certain distance threshold
    :param sim: Simulator object
    :param sensor_name: name of the visual sensor to be used for raycasting
    :param crosshair_pos: 2D coordiante in the viewport towards which the
        ray will be cast
    :param max_distance: distance threshold beyond which objects won't
        be considered
    """
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera
    center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))

    raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)

    closest_object = -1
    closest_dist = 1000.0
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            if hit.ray_distance < closest_dist:
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object


# %%
# Test the raycast utility.

with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects"))
    )
    obj_path = "test_assets/objects/chair"
    chair_template_id = obj_attr_mgr.load_object_configs(
        str(os.path.join(data_path, obj_path))
    )[0]
    chair_attr = obj_attr_mgr.get_template_by_ID(chair_template_id)
    obj_attr_mgr.register_template(chair_attr)
    object_id = sim.add_object_by_handle(chair_attr.handle)
    print(f"Chair's object id is {object_id}")

    set_object_in_front_of_agent(sim, object_id, -1.5)
    sim.set_object_motion_type(
        habitat_sim.physics.MotionType.STATIC, object_id
    )
    if make_video:
        # Visualize the agent's initial position
        simulate_and_make_vid(
            sim, [190, 128], "sim-before-grab", dt=1.0, open_vid=show_video
        )

    # Distance threshold=2 is greater than agent-to-chair distance.
    # Should return chair's object id
    closest_object = raycast(
        sim, "rgb", crosshair_pos=[128, 190], max_distance=2.0
    )
    print(f"Closest Object ID: {closest_object} using 2.0 threshold")
    assert (
        closest_object == object_id
    ), f"Could not pick chair with ID: {object_id}"

    # Distance threshold=1 is smaller than agent-to-chair distance .
    # Should return -1
    closest_object = raycast(
        sim, "rgb", crosshair_pos=[128, 190], max_distance=1.0
    )
    print(f"Closest Object ID: {closest_object} using 1.0 threshold")
    assert closest_object == -1, "Agent shoud not be able to pick any object"


# %%
# @title Define a Grab/Release action and create a new action space.
# @markdown Each new action is defined by a `ActionSpec` and an `ActuationSpec`. `ActionSpec` is mapping between the action name and its corresponding `ActuationSpec`. `ActuationSpec` contains all the necessary specifications required to define the action.

from habitat.config.default import _C, CN
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim.physics import MotionType


# @markdown For instance, `GrabReleaseActuationSpec` contains the following:
# @markdown - `visual_sensor_name` defines which viewport (rgb, depth, etc) to to use to cast the ray.
# @markdown - `crosshair_pos` stores the position in the viewport through which the ray passes. Any object which intersects with this ray can be grabbed by the agent.
# @markdown - `amount` defines a distance threshold. Objects which are farther than the treshold cannot be picked up by the agent.
@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 2.0


# @markdown Then, we extend the `HabitatSimV1ActionSpaceConfiguration` to add the above action into the agent's action space. `ActionSpaceConfiguration` is a mapping between action name and the corresponding `ActionSpec`
@registry.register_action_space_configuration(name="RearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            )
        }

        config.update(new_config)

        return config


# @markdown Finally, we extend `SimualtorTaskAction` which tells the simulator which action to call when a named action ('GRAB_RELEASE' in this case) is predicte by the agent's policy.
@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)


_C.TASK.ACTIONS.GRAB_RELEASE = CN()
_C.TASK.ACTIONS.GRAB_RELEASE.TYPE = "GrabOrReleaseAction"
_C.SIMULATOR.CROSSHAIR_POS = [128, 160]
_C.SIMULATOR.GRAB_DISTANCE = 2.0
_C.SIMULATOR.VISUAL_SENSOR = "rgb"

# %% [markdown]
# ##Setup Simulator Class for Rearrangement Task
#
# ![sim](https://drive.google.com/uc?id=1ce6Ti-gpumMEyfomqAKWqOspXm6tN4_8)

# %%
# @title RearrangementSim Class
# @markdown Here we will extend the `HabitatSim` class for the rearrangement task. We will make the following changes:
# @markdown - define a new `_initialize_objects` function which will load the object in its initial configuration as defined by the episode.
# @markdown - define a `gripped_object_id` property that stores whether the agent is holding any object or not.
# @markdown - modify the `step` function of the simulator to use the `grab/release` action we define earlier.

# @markdown #### Writing the `step` function:
# @markdown Since we added a new action for this task, we have to modify the `step` function to define what happens when `grab/release` action is called. If a simple navigation action (`move_forward`, `turn_left`, `turn_right`) is called, we pass it forward to `act` function of the agent which already defines the behavior of these actions.

# @markdown For the `grab/release` action, if the agent is not already holding an object, we first call the `raycast` function using the values from the `ActuationSpec` to see if any object is grippable. If it returns a valid object id, we put the object in a "invisible" inventory and remove it from the scene.

# @markdown If the agent was already holding an object, `grab/release` action will try release the object at the same relative position as it was grabbed. If the object can be placed without any collision, then the `release` action is successful.

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum


@registry.register_simulator(name="RearrangementSim-v0")
class RearrangementSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim with
    object rearrangement functionalities.
    """

    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        self.grip_offset = np.eye(4)

        agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_config = self._get_agent_config(agent_id)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        self._initialize_objects()

    def reset(self):
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def _initialize_objects(self):
        objects = self.habitat_config.objects[0]
        obj_attr_mgr = self.get_object_template_manager()
        obj_attr_mgr.load_configs(
            str(os.path.join(data_path, "test_assets/objects"))
        )
        # first remove all existing objects
        existing_object_ids = self.get_existing_object_ids()

        if len(existing_object_ids) > 0:
            for obj_id in existing_object_ids:
                self.remove_object(obj_id)

        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

        if objects is not None:
            object_template = objects["object_template"]
            object_pos = objects["position"]
            object_rot = objects["rotation"]

            object_template_id = obj_attr_mgr.load_object_configs(
                object_template
            )[0]
            object_attr = obj_attr_mgr.get_template_by_ID(object_template_id)
            obj_attr_mgr.register_template(object_attr)

            object_id = self.add_object_by_handle(object_attr.handle)
            self.sim_object_to_objid_mapping[object_id] = objects["object_id"]
            self.objid_to_sim_object_mapping[objects["object_id"]] = object_id

            self.set_translation(object_pos, object_id)
            if isinstance(object_rot, list):
                object_rot = quat_from_coeffs(object_rot)

            object_rot = quat_to_magnum(object_rot)
            self.set_rotation(object_rot, object_id)

            self.set_object_motion_type(MotionType.STATIC, object_id)

        # Recompute the navmesh after placing all the objects.
        self.recompute_navmesh(self.pathfinder, self.navmesh_settings, True)

    def _sync_gripped_object(self, gripped_object_id):
        r"""
        Sync the gripped object with the object associated with the agent.
        """
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._default_agent.scene_node.transformation
            )
            self.set_transformation(
                agent_body_transformation, gripped_object_id
            )
            translation = agent_body_transformation.transform_point(
                np.array([0, 2.0, 0])
            )
            self.set_translation(translation, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def step(self, action: int):
        dt = 1 / 60.0
        self._num_total_frames += 1
        collided = False
        gripped_object_id = self.gripped_object_id

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        if action_spec.name == "grab_or_release_object_under_crosshair":
            # If already holding an agent
            if gripped_object_id != -1:
                agent_body_transformation = (
                    self._default_agent.scene_node.transformation
                )
                T = np.dot(agent_body_transformation, self.grip_offset)

                self.set_transformation(T, gripped_object_id)

                position = self.get_translation(gripped_object_id)

                if self.pathfinder.is_navigable(position):
                    self.set_object_motion_type(
                        MotionType.STATIC, gripped_object_id
                    )
                    gripped_object_id = -1
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )
            # if not holding an object, then try to grab
            else:
                gripped_object_id = raycast(
                    self,
                    action_spec.actuation.visual_sensor_name,
                    crosshair_pos=action_spec.actuation.crosshair_pos,
                    max_distance=action_spec.actuation.amount,
                )

                # found a grabbable object.
                if gripped_object_id != -1:
                    agent_body_transformation = (
                        self._default_agent.scene_node.transformation
                    )

                    self.grip_offset = np.dot(
                        np.array(agent_body_transformation.inverted()),
                        np.array(self.get_transformation(gripped_object_id)),
                    )
                    self.set_object_motion_type(
                        MotionType.KINEMATIC, gripped_object_id
                    )
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )

        else:
            collided = self._default_agent.act(action)
            self._last_state = self._default_agent.get_state()

        # step physics by dt
        super().step_world(dt)

        # Sync the gripped object after the agent moves.
        self._sync_gripped_object(gripped_object_id)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations


# %% [markdown]
# ## Create the Rearrangement Task
# ![task](https://drive.google.com/uc?id=1N75Mmi6aigh33uL765ljsAqLzFmcs7Zn)

# %%
# @title Implement new sensors and measurements
# @markdown After defining the dataset, action space and simulator functions for the rearrangement task, we are one step closer to training agents to solve this task.

# @markdown Here we define inputs to the policy and other measurements required to design reward functions.

# @markdown **Sensors**: These define various part of the simulator state that's visible to the agent. For simplicity, we'll assume that agent knows the object's current position, object's final goal position relative to the agent's current position.
# @markdown - Object's current position will be made given by the `ObjectPosition` sensor
# @markdown - Object's goal position will be available through the `ObjectGoal` sensor.
# @markdown - Finally, we will also use `GrippedObject` sensor to tell the agent if it's holding any object or not.

# @markdown **Measures**: These define various metrics about the task which can be used to measure task progress and define rewards. Note that measurements are *privileged* information not accessible to the agent as part of the observation space. We will need the following measurements:
# @markdown - `AgentToObjectDistance` which measure the euclidean distance between the agent and the object.
# @markdown - `ObjectToGoalDistance` which measures the euclidean distance between the object and the goal.

from gym import spaces

import habitat_sim
from habitat.config.default import CN, Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import PointGoalSensor


@registry.register_sensor
class GrippedObjectSensor(Sensor):
    cls_uuid = "gripped_object_id"

    def __init__(
        self, *args: Any, sim: RearrangementSim, config: Config, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):

        return spaces.Discrete(len(self._sim.get_existing_object_ids()))

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: Episode,
        *args: Any,
        **kwargs: Any,
    ):
        obj_id = self._sim.sim_object_to_objid_mapping.get(
            self._sim.gripped_object_id, -1
        )
        return obj_id


@registry.register_sensor
class ObjectPosition(PointGoalSensor):
    cls_uuid: str = "object_position"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        object_id = self._sim.get_existing_object_ids()[0]
        object_position = self._sim.get_translation(object_id)
        pointgoal = self._compute_pointgoal(
            agent_position, rotation_world_agent, object_position
        )
        return pointgoal


@registry.register_sensor
class ObjectGoal(PointGoalSensor):
    cls_uuid: str = "object_goal"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        goal_position = np.array(episode.goals.position, dtype=np.float32)

        point_goal = self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
        return point_goal


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_to_goal_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        sim_obj_id = self._sim.get_existing_object_ids()[0]

        previous_position = np.array(
            self._sim.get_translation(sim_obj_id)
        ).tolist()
        goal_position = episode.goals.position
        self._metric = self._euclidean_distance(
            previous_position, goal_position
        )


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        sim_obj_id = self._sim.get_existing_object_ids()[0]
        previous_position = np.array(
            self._sim.get_translation(sim_obj_id)
        ).tolist()

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position

        self._metric = self._euclidean_distance(
            previous_position, agent_position
        )


# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK GRIPPED OBJECT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GRIPPED_OBJECT_SENSOR = CN()
_C.TASK.GRIPPED_OBJECT_SENSOR.TYPE = "GrippedObjectSensor"
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT POSITIONS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_POSITION = CN()
_C.TASK.OBJECT_POSITION.TYPE = "ObjectPosition"
_C.TASK.OBJECT_POSITION.GOAL_FORMAT = "POLAR"
_C.TASK.OBJECT_POSITION.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT GOALS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_GOAL = CN()
_C.TASK.OBJECT_GOAL.TYPE = "ObjectGoal"
_C.TASK.OBJECT_GOAL.GOAL_FORMAT = "POLAR"
_C.TASK.OBJECT_GOAL.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # OBJECT_DISTANCE_TO_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_TO_GOAL_DISTANCE = CN()
_C.TASK.OBJECT_TO_GOAL_DISTANCE.TYPE = "ObjectToGoalDistance"
# -----------------------------------------------------------------------------
# # OBJECT_DISTANCE_FROM_AGENT MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.AGENT_TO_OBJECT_DISTANCE = CN()
_C.TASK.AGENT_TO_OBJECT_DISTANCE.TYPE = "AgentToObjectDistance"

from habitat.config.default import CN, Config

# %%
# @title Define `RearrangementTask` by extending `NavigationTask`
from habitat.tasks.nav.nav import NavigationTask, merge_sim_episode_config


def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()
    sim_config.objects = [episode.objects.__dict__]
    sim_config.freeze()

    return sim_config


@registry.register_task(name="RearrangementTask-v0")
class RearrangementTask(NavigationTask):
    r"""Embodied Rearrangement Task
    Goal: An agent must place objects at their corresponding goal position.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)


# %% [markdown]
# ## Implement a hard-coded and an RL agent
#
#

# %%
# @title Load the `RearrangementTask` in Habitat-Lab and run a hard-coded agent
import habitat

config = habitat.get_config("configs/tasks/pointnav.yaml")
config.defrost()
config.ENVIRONMENT.MAX_EPISODE_STEPS = 50
config.SIMULATOR.TYPE = "RearrangementSim-v0"
config.SIMULATOR.ACTION_SPACE_CONFIG = "RearrangementActions-v0"
config.SIMULATOR.GRAB_DISTANCE = 2.0
config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
config.TASK.TYPE = "RearrangementTask-v0"
config.TASK.SUCCESS_DISTANCE = 1.0
config.TASK.SENSORS = [
    "GRIPPED_OBJECT_SENSOR",
    "OBJECT_POSITION",
    "OBJECT_GOAL",
]
config.TASK.GOAL_SENSOR_UUID = "object_goal"
config.TASK.MEASUREMENTS = [
    "OBJECT_TO_GOAL_DISTANCE",
    "AGENT_TO_OBJECT_DISTANCE",
]
config.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "GRAB_RELEASE"]
config.DATASET.TYPE = "RearrangementDataset-v0"
config.DATASET.SPLIT = "train"
config.DATASET.DATA_PATH = (
    "data/datasets/rearrangement/coda/v1/{split}/{split}.json.gz"
)
config.freeze()


def print_info(obs, metrics):
    print(
        "Gripped Object: {}, Distance To Object: {}, Distance To Goal: {}".format(
            obs["gripped_object_id"],
            metrics["agent_to_object_distance"],
            metrics["object_to_goal_distance"],
        )
    )


try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass

with habitat.Env(config) as env:
    obs = env.reset()
    obs_list = []
    # Get closer to the object
    while True:
        obs = env.step(1)
        obs_list.append(obs)
        metrics = env.get_metrics()
        print_info(obs, metrics)
        if metrics["agent_to_object_distance"] < 2.0:
            break

    # Grab the object
    obs = env.step(2)
    obs_list.append(obs)
    metrics = env.get_metrics()
    print_info(obs, metrics)
    assert obs["gripped_object_id"] != -1

    # Get closer to the goal
    while True:
        obs = env.step(1)
        obs_list.append(obs)
        metrics = env.get_metrics()
        print_info(obs, metrics)
        if metrics["object_to_goal_distance"] < 2.0:
            break

    # Release the object
    obs = env.step(2)
    obs_list.append(obs)
    metrics = env.get_metrics()
    print_info(obs, metrics)
    assert obs["gripped_object_id"] == -1

    if make_video:
        make_video_cv2(
            obs_list,
            [190, 128],
            "hard-coded-agent",
            fps=5.0,
            open_vid=show_video,
        )

# %%
# @title Create a task specific RL Environment with a new reward definition.
# @markdown We create a `RearragenmentRLEnv` class and modify the `get_reward()` function.
# @markdown The reward sturcture is as follows:
# @markdown - The agent gets a positive reward if the agent gets closer to the object otherwise a negative reward.
# @markdown - The agent gets a positive reward if it moves the object closer to goal otherwise a negative reward.
# @markdown - The agent gets a positive reward when the agent "picks" up an object for the first time. For all other "grab/release" action, it gets a negative reward.
# @markdown - The agent gets a slack penalty of -0.01 for every action it takes in the environment.
# @markdown - Finally the agent gets a large success reward when the episode is completed successfully.

from typing import Optional, Type

import numpy as np

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv


@baseline_registry.register_env(name="RearrangementRLEnv")
class RearrangementRLEnv(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._prev_measure = {
            "agent_to_object_distance": 0.0,
            "object_to_goal_distance": 0.0,
            "gripped_object_id": -1,
            "gripped_object_count": 0,
        }

        super().__init__(config, dataset)

        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE

    def reset(self):
        self._previous_action = None
        observations = super().reset()

        self._prev_measure.update(self.habitat_env.get_metrics())
        self._prev_measure["gripped_object_id"] = -1
        self._prev_measure["gripped_object_count"] = 0

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD
        gripped_success_reward = 0.0
        episode_success_reward = 0.0
        agent_to_object_dist_reward = 0.0
        object_to_goal_dist_reward = 0.0

        action_name = self._env.task.get_action_name(
            self._previous_action["action"]
        )

        # If object grabbed, add a success reward
        # The reward gets awarded only once for an object.
        if (
            action_name == "GRAB_RELEASE"
            and observations["gripped_object_id"] >= 0
        ):
            obj_id = observations["gripped_object_id"]
            self._prev_measure["gripped_object_count"] += 1

            gripped_success_reward = (
                self._rl_config.GRIPPED_SUCCESS_REWARD
                if self._prev_measure["gripped_object_count"] == 1
                else 0.0
            )
        # add a penalty everytime grab/action is called and doesn't do anything
        elif action_name == "GRAB_RELEASE":
            gripped_success_reward += -0.1

        self._prev_measure["gripped_object_id"] = observations[
            "gripped_object_id"
        ]

        # If the action is not a grab/release action, and the agent
        # has not picked up an object, then give reward based on agent to
        # object distance.
        if (
            action_name != "GRAB_RELEASE"
            and self._prev_measure["gripped_object_id"] == -1
        ):
            agent_to_object_dist_reward = self.get_agent_to_object_dist_reward(
                observations
            )

        # If the action is not a grab/release action, and the agent
        # has picked up an object, then give reward based on object to
        # to goal distance.
        if (
            action_name != "GRAB_RELEASE"
            and self._prev_measure["gripped_object_id"] != -1
        ):
            object_to_goal_dist_reward = self.get_object_to_goal_dist_reward()

        if (
            self._episode_success(observations)
            and self._prev_measure["gripped_object_id"] == -1
            and action_name == "STOP"
        ):
            episode_success_reward = self._rl_config.SUCCESS_REWARD

        reward += (
            agent_to_object_dist_reward
            + object_to_goal_dist_reward
            + gripped_success_reward
            + episode_success_reward
        )

        return reward

    def get_agent_to_object_dist_reward(self, observations):
        """
        Encourage the agent to move towards the closest object which is not already in place.
        """
        curr_metric = self._env.get_metrics()["agent_to_object_distance"]
        prev_metric = self._prev_measure["agent_to_object_distance"]
        dist_reward = prev_metric - curr_metric

        self._prev_measure["agent_to_object_distance"] = curr_metric

        return dist_reward

    def get_object_to_goal_dist_reward(self):
        curr_metric = self._env.get_metrics()["object_to_goal_distance"]
        prev_metric = self._prev_measure["object_to_goal_distance"]
        dist_reward = prev_metric - curr_metric

        self._prev_measure["object_to_goal_distance"] = curr_metric

        return dist_reward

    def _episode_success(self, observations):
        r"""Returns True if object is within distance threshold of the goal."""
        dist = self._env.get_metrics()["object_to_goal_distance"]
        if (
            abs(dist) > self._success_distance
            or observations["gripped_object_id"] != -1
        ):
            return False
        return True

    def _gripped_success(self, observations):
        if (
            observations["gripped_object_id"] >= 0
            and observations["gripped_object_id"]
            != self._prev_measure["gripped_object_id"]
        ):
            return True

        return False

    def get_done(self, observations):
        done = False
        action_name = self._env.task.get_action_name(
            self._previous_action["action"]
        )
        if self._env.episode_over or (
            self._episode_success(observations)
            and self._prev_measure["gripped_object_id"] == -1
            and action_name == "STOP"
        ):
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["episode_success"] = self._episode_success(observations)
        return info


# %%
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.policy import Net, Policy
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.env_utils import make_env_fn


def construct_envs(
    config,
    env_class,
    workers_ignore_signals=False,
):
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_processes = config.NUM_ENVIRONMENTS
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = habitat.datasets.make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            scenes = scenes * num_processes

        random.shuffle(scenes)

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    envs = habitat.ThreadedVectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


class RearrangementBaselinePolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__(
            RearrangementBaselineNet(
                observation_space=observation_space, hidden_size=hidden_size
            ),
            action_space.n,
        )

    def from_config(cls, config, envs):
        pass


class RearrangementBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        self._n_input_goal = observation_space.spaces[
            ObjectGoal.cls_uuid
        ].shape[0]

        self._hidden_size = hidden_size

        self.state_encoder = build_rnn_state_encoder(
            2 * self._n_input_goal, self._hidden_size
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        object_goal_encoding = observations[ObjectGoal.cls_uuid]
        object_pos_encoding = observations[ObjectPosition.cls_uuid]

        x = [object_goal_encoding, object_pos_encoding]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


@baseline_registry.register_trainer(name="ppo-rearrangement")
class RearrangementTrainer(PPOTrainer):
    supported_tasks = ["RearrangementTask-v0"]

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = RearrangementBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("This trainer does not support distributed")
        self._init_train()

        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda _: 1 - self.percent_done(),
        )
        ppo_cfg = self.config.RL.PPO

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while not self.is_done():

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                count_steps_delta = 0
                for _step in range(ppo_cfg.num_steps):
                    count_steps_delta += self._collect_rollout_step()

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss),
                    count_steps_delta,
                )
                self.num_updates_done += 1

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in self.window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward",
                    deltas["reward"] / deltas["count"],
                    self.num_steps_done,
                )

                # Check to see if there are any metrics
                # that haven't been logged yet

                for k, v in deltas.items():
                    if k not in {"reward", "count"}:
                        writer.add_scalar(
                            "metric/" + k,
                            v / deltas["count"],
                            self.num_steps_done,
                        )

                losses = [value_loss, action_loss]
                for l, k in zip(losses, ["value, policy"]):
                    writer.add_scalar("losses/" + k, l, self.num_steps_done)

                # log stats
                if self.num_updates_done % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            self.num_updates_done,
                            self.num_steps_done / (time.time() - self.t_start),
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            self.num_updates_done,
                            self.env_time,
                            self.pth_time,
                            self.num_steps_done,
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(self.window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(step=self.num_steps_done),
                    )
                    count_checkpoints += 1

            self.envs.close()

    def eval(self) -> None:
        r"""Evaluates the current model
        Returns:
            None
        """

        config = self.config.clone()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.NUM_ENVIRONMENTS = 1
            config.freeze()

        logger.info(f"env config: {config}")
        with construct_envs(config, get_env_class(config.ENV_NAME)) as envs:
            observations = envs.reset()
            batch = batch_obs(observations, device=self.device)

            current_episode_reward = torch.zeros(
                envs.num_envs, 1, device=self.device
            )
            ppo_cfg = self.config.RL.PPO
            test_recurrent_hidden_states = torch.zeros(
                config.NUM_ENVIRONMENTS,
                self.actor_critic.net.num_recurrent_layers,
                ppo_cfg.hidden_size,
                device=self.device,
            )
            prev_actions = torch.zeros(
                config.NUM_ENVIRONMENTS,
                1,
                device=self.device,
                dtype=torch.long,
            )
            not_done_masks = torch.zeros(
                config.NUM_ENVIRONMENTS,
                1,
                device=self.device,
                dtype=torch.bool,
            )

            rgb_frames = [
                [] for _ in range(self.config.NUM_ENVIRONMENTS)
            ]  # type: List[List[np.ndarray]]

            if len(config.VIDEO_OPTION) > 0:
                os.makedirs(config.VIDEO_DIR, exist_ok=True)

            self.actor_critic.eval()

            for _i in range(config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS):
                current_episodes = envs.current_episodes()

                with torch.no_grad():
                    (
                        _,
                        actions,
                        _,
                        test_recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )

                    prev_actions.copy_(actions)

                outputs = envs.step([a[0].item() for a in actions])

                observations, rewards, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
                batch = batch_obs(observations, device=self.device)

                not_done_masks = torch.tensor(
                    [[not done] for done in dones],
                    dtype=torch.bool,
                    device="cpu",
                )

                rewards = torch.tensor(
                    rewards, dtype=torch.float, device=self.device
                ).unsqueeze(1)

                current_episode_reward += rewards

                # episode ended
                if not not_done_masks[0].item():
                    generate_video(
                        video_option=self.config.VIDEO_OPTION,
                        video_dir=self.config.VIDEO_DIR,
                        images=rgb_frames[0],
                        episode_id=current_episodes[0].episode_id,
                        checkpoint_idx=0,
                        metrics=self._extract_scalars_from_info(infos[0]),
                        tb_writer=None,
                    )

                    print("Evaluation Finished.")
                    print("Success: {}".format(infos[0]["episode_success"]))
                    print(
                        "Reward: {}".format(current_episode_reward[0].item())
                    )
                    print(
                        "Distance To Goal: {}".format(
                            infos[0]["object_to_goal_distance"]
                        )
                    )

                    return

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[0], infos[0])
                    rgb_frames[0].append(frame)

                not_done_masks = not_done_masks.to(device=self.device)


# %%
# %load_ext tensorboard
# %tensorboard --logdir data/tb

# %%
# @title Train an RL agent on a single episode
# !if [ -d "data/tb" ]; then rm -r data/tb; fi

import random

import numpy as np
import torch

import habitat
from habitat import Config
from habitat_baselines.config.default import get_config as get_baseline_config

baseline_config = get_baseline_config(
    "habitat_baselines/config/pointnav/ppo_pointnav.yaml"
)
baseline_config.defrost()

baseline_config.TASK_CONFIG = config
baseline_config.TRAINER_NAME = "ddppo"
baseline_config.ENV_NAME = "RearrangementRLEnv"
baseline_config.SIMULATOR_GPU_ID = 0
baseline_config.TORCH_GPU_ID = 0
baseline_config.VIDEO_OPTION = ["disk"]
baseline_config.TENSORBOARD_DIR = "data/tb"
baseline_config.VIDEO_DIR = "data/videos"
baseline_config.NUM_ENVIRONMENTS = 2
baseline_config.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
baseline_config.CHECKPOINT_FOLDER = "data/checkpoints"
baseline_config.TOTAL_NUM_STEPS = -1.0

if vut.is_notebook():
    baseline_config.NUM_UPDATES = 400  # @param {type:"number"}
else:
    baseline_config.NUM_UPDATES = 1

baseline_config.LOG_INTERVAL = 10
baseline_config.NUM_CHECKPOINTS = 5
baseline_config.LOG_FILE = "data/checkpoints/train.log"
baseline_config.EVAL.SPLIT = "train"
baseline_config.RL.SUCCESS_REWARD = 2.5  # @param {type:"number"}
baseline_config.RL.SUCCESS_MEASURE = "object_to_goal_distance"
baseline_config.RL.REWARD_MEASURE = "object_to_goal_distance"
baseline_config.RL.GRIPPED_SUCCESS_REWARD = 2.5  # @param {type:"number"}

baseline_config.freeze()
random.seed(baseline_config.TASK_CONFIG.SEED)
np.random.seed(baseline_config.TASK_CONFIG.SEED)
torch.manual_seed(baseline_config.TASK_CONFIG.SEED)

if __name__ == "__main__":
    trainer = RearrangementTrainer(baseline_config)
    trainer.train()
    trainer.eval()

    if make_video:
        video_file = os.listdir("data/videos")[0]
        vut.display_video(os.path.join("data/videos", video_file))
