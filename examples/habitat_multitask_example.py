from argparse import ArgumentParser

import cv2

from habitat.config import get_config
from habitat.core.env import MultiTaskEnv
from habitat.tasks.nav.nav import NavigationEpisode

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"


def rgb2bgr(image):
    return image[..., ::-1]


def visualize(flag: bool, obs):
    if flag:
        cv2.imshow("RGB", rgb2bgr(obs["rgb"]))


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "-i",
        "--interactive",
        help="Run demo interactively",
        action="store_true",
    )
    args = args.parse_args()
    ### One Env, many tasks ###
    # cfg = get_config('pointnav.yaml')
    cfg = get_config("configs/test/habitat_multitask_example.yaml")
    # cfg.defrost()
    # cfg.TASKS[0].SENSORS = ["PROXIMITY_SENSOR"]
    # cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        print(
            "{} episodes created from config file".format(len(env._episodes))
        )
        scene_sort_keys = {}
        for e in env.episodes:
            if e.scene_id not in scene_sort_keys:
                scene_sort_keys[e.scene_id] = len(scene_sort_keys)
        print("Number of scenes", scene_sort_keys)
        # usual OpenAI Gym-like env-agent loop
        n_episodes = 2
        print(env._tasks)
        taks_id = 0
        for _ in range(n_episodes):
            obs = env.reset()
            visualize(args.interactive, obs)
            actions = 0
            print(
                "Current task is {} with id {} and task label {}, with scene id {}".format(
                    env.task.__class__.__name__,
                    env._curr_task_idx,
                    obs["task_label"],
                    env.current_episode.scene_id,
                )
            )

            # make sure we sample a different episode even if same task is passed
            # env._config.defrost()
            # env._config.SEED = i
            # env._config.freeze()
            while not env.episode_over:
                print("Current obs space", env.observation_space)
                print("Current action space", env.action_space)
                if isinstance(env.current_episode, NavigationEpisode):
                    print("Goal:", env.current_episode.goals)
                if args.interactive:
                    keystroke = cv2.waitKey(0)
                    # ord gets unicode from one-char string
                    if keystroke == ord(FORWARD_KEY):
                        action = "MOVE_FORWARD"
                    elif keystroke == ord(LEFT_KEY):
                        action = "TURN_LEFT"
                    elif keystroke == ord(RIGHT_KEY):
                        action = "TURN_RIGHT"
                    elif keystroke == ord(FINISH):
                        action = "STOP"
                    else:
                        print("INVALID KEY")
                        continue
                    if action not in env.task.actions:
                        print("Invalid action!")
                        continue

                else:
                    # execute random action
                    action = env.action_space.sample()
                obs = env.step(action)
                actions += 1
                visualize(args.interactive, obs)

            print("Episode finished after {} steps.".format(actions))
