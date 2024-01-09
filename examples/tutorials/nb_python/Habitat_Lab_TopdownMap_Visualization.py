# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     provenance: []
#   gpuClass: standard
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
# # Habitat Lab: Topdown Map Visualization
#
# ## Initial setup and imports:

# %%
# [setup]
import os
from typing import TYPE_CHECKING, Union, cast

import git
import matplotlib.pyplot as plt
import numpy as np

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(
    dir_path, "examples/tutorials/habitat_lab_visualization/"
)
os.makedirs(output_path, exist_ok=True)
os.chdir(dir_path)
# [/setup]


# %% [markdown]
# ### Download (testing) 3D scenes:

# %%
# !python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path {data_path} --no-replace

# %% [markdown]
# ### Download point-goal navigation episodes for the test scenes:

# %%
# !python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path {data_path} --no-replace


# %%
# [example_1]
def example_pointnav_draw_target_birdseye_view():
    # Define NavigationEpisode parameters
    goal_radius = 0.5
    goal = NavigationGoal(position=[10, 0.25, 10], radius=goal_radius)
    agent_position = [0, 0.25, 0]
    agent_rotation = -np.pi / 4

    # Create dummy episode for birdseye view visualization
    dummy_episode = NavigationEpisode(
        goals=[goal],
        episode_id="dummy_id",
        scene_id="dummy_scene",
        start_position=agent_position,
        start_rotation=agent_rotation,  # type: ignore[arg-type]
    )

    agent_position = np.array(agent_position)
    # Draw birdseye view
    target_image = maps.pointnav_draw_target_birdseye_view(
        agent_position,
        agent_rotation,
        np.asarray(dummy_episode.goals[0].position),
        goal_radius=dummy_episode.goals[0].radius,
        agent_radius_px=25,
    )
    plt.imshow(target_image)
    plt.title("pointnav_target_image.png")
    plt.show()


# [/example_1]


# %%
# [example_2]
def example_pointnav_draw_target_birdseye_view_agent_on_border():
    # Define NavigationGoal
    goal_radius = 0.5
    goal = NavigationGoal(position=[0, 0.25, 0], radius=goal_radius)
    # For defined goal create 4 NavigationEpisodes
    # with agent being placed on different borders,
    # draw birdseye view for each episode and save image to disk
    ii = 0
    for x_edge in [-1, 0, 1]:
        for y_edge in [-1, 0, 1]:
            if not np.bitwise_xor(x_edge == 0, y_edge == 0):
                continue
            ii += 1
            agent_position = [7.8 * x_edge, 0.25, 7.8 * y_edge]
            agent_rotation = np.pi / 2

            dummy_episode = NavigationEpisode(
                goals=[goal],
                episode_id="dummy_id",
                scene_id="dummy_scene",
                start_position=agent_position,
                start_rotation=agent_rotation,  # type: ignore[arg-type]
            )

            agent_position = np.array(agent_position)
            target_image = maps.pointnav_draw_target_birdseye_view(
                agent_position,
                agent_rotation,
                np.asarray(dummy_episode.goals[0].position),
                goal_radius=dummy_episode.goals[0].radius,
                agent_radius_px=25,
            )
            plt.imshow(target_image)
            plt.title("pointnav_target_image_edge_%d.png" % ii)
            plt.show()


# [/example_2]


# %%
# [example_3]
def example_get_topdown_map():
    # Create habitat config
    config = habitat.get_config(
        config_path=os.path.join(
            dir_path,
            "habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        )
    )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Load the first episode
        env.reset()
        # Generate topdown map
        top_down_map = maps.get_topdown_map_from_sim(
            cast("HabitatSim", env.sim), map_resolution=1024
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        # By default, `get_topdown_map_from_sim` returns image
        # containing 0 if occupied, 1 if unoccupied, and 2 if border
        # The line below recolors returned image so that
        # occupied regions are colored in [255, 255, 255],
        # unoccupied in [128, 128, 128] and border is [0, 0, 0]
        top_down_map = recolor_map[top_down_map]
        plt.imshow(top_down_map)
        plt.title("top_down_map.png")
        plt.show()


# [/example_3]


# %%
# [example_4]
class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


def example_top_down_map_measure():
    # Create habitat config
    config = habitat.get_config(
        config_path=os.path.join(
            dir_path,
            "habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        )
    )
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )
        # Create video of agent navigating in the first episode
        num_episodes = 1
        for _ in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(observations, info)

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations)
                if action is None:
                    break

                # Step in the environment
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            current_episode = env.current_episode
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            # Create video from images and save to disk
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()
            # Display video
            vut.display_video(f"{output_path}/{video_name}.mp4")


# [/example_4]


# %%
if __name__ == "__main__":
    example_pointnav_draw_target_birdseye_view()
    example_pointnav_draw_target_birdseye_view_agent_on_border()
    example_get_topdown_map()
    example_top_down_map_measure()
