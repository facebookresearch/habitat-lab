from collections import defaultdict
import pickle
import os

from omegaconf import OmegaConf
import tqdm
import cv2
import habitat


def get_config(split: str, shuffle: bool = True):
    config = habitat.get_config(
        "habitat-lab/habitat-lab/habitat/config/benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml"
    )
    OmegaConf.set_readonly(config, False)
    # for speed: remove/reduce unused sensors
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 1
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 1
    config.habitat.environment.iterator_options.shuffle = shuffle
    config.habitat.dataset.split = split
    del config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    del config.habitat.task.lab_sensors.instance_imagegoal_hfov_sensor
    del config.habitat.task.lab_sensors.compass_sensor
    del config.habitat.task.lab_sensors.gps_sensor
    OmegaConf.set_readonly(config, True)
    return config


def dd_list():
    """necessary for pickle"""
    return defaultdict(list)


def main(
    split: str = "minival",
    shuffle: bool = True,
    save_first: int = -1,
    save_img: bool = False,
    save_dir: str = "goal_images",
) -> None:
    config = get_config(split, shuffle)
    print(OmegaConf.to_yaml(config))

    env = habitat.Env(config=config)

    # scene_id -> instance_id -> list of image goals
    image_goal_dataset = defaultdict(dd_list)

    num_eps = min(save_first, env.number_of_episodes)
    if num_eps == -1:
        num_eps = env.number_of_episodes
    if save_img:
        os.makedirs(save_dir, exist_ok=True)

    for _ in tqdm.trange(num_eps):
        obs = env.reset()
        ep = env.current_episode
        image_goal_dataset[ep.scene_id][ep.goal_object_id] = obs[
            "instance_imagegoal"
        ]
        if save_img:
            short_scene_id = ep.scene_id.split("/")[-1].split(".")[0]
            save_path = os.path.join(
                save_dir,
                f"{short_scene_id}_{ep.goal_object_id}_{ep.goal_image_id}.png",
            )
            img = cv2.cvtColor(obs["instance_imagegoal"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img)

    with open("image_goal_dataset.pkl", "wb") as f:
        pickle.dump(image_goal_dataset, f)


if __name__ == "__main__":
    main(split="val", shuffle=False, save_first=100, save_img=True)
