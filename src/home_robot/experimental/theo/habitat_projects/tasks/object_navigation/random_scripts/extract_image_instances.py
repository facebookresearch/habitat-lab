"""
This script is intended to run from the "src" root:
python home_robot/experimental/theo/habitat_projects/tasks/object_navigation/random_scripts/extract_image_instances.py
"""
import habitat
from habitat.core.env import Env


if __name__ == "__main__":
    # config_path = "home_robot/experimental/theo/habitat_projects/tasks/object_navigation/configs/task/hm3d_imageinstancegoal_val.yaml"
    config_path = "habitat/config/benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml"
    config = habitat.get_config(config_path)
    env = Env(config=config)
    obs = env.reset()
    scene_id = env.current_episode.scene_id.split("/")[-1].split(".")[0]
    episode_id = env.current_episode.episode_id
    print(obs["instance_imagegoal"])
    print(scene_id)
    print(episode_id)
