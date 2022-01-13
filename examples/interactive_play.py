#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the robot to interact with the environment. Run as
```
python examples/interative_play.py
```

To Run you need PyGame installed.

By default this controls with velocity control (which makes controlling the
robot hard). To use IK control instead: run with
```
python examples/interactive_play.py --cfg configs/tasks/rearrangepick_replica_cad_example_ik.yaml
```

Controls:
- For velocity control
    - 1-7 to increase the motor target for the robot arm joints
    - Q-U to decrease the motor target for the robot arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the robot base around
- PERIOD to print the current world coordinates of the robot base.

Change the grip type:
- Suction gripper `TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER "SuctionGraspAction"`

Record and play back trajectories:
- To record a trajectory add `--save-actions --save-actions-count 200` to
  record a truncated episode length of 200.
- By default the trajectories are saved to data/interactive_play_replays/play_actions.txt
- Play the trajectories back with `--load-actions data/interactive_play_replays/play_actions.txt`
"""

import argparse
import os
import os.path as osp
import time

import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.tasks.rearrange.actions import ArmEEAction
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.render_wrapper import overlay_frame
from habitat_sim.utils import viz_utils as vut

try:
    import pygame
except ImportError:
    pygame = None

DEFAULT_CFG = "configs/tasks/rearrange/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


def step_env(env, action_name, action_args, args):
    return env.step({"action": action_name, "action_args": action_args})


def get_input_vel_ctlr(skip_pygame, arm_action, g_args, prev_obs, env):
    if skip_pygame:
        return step_env(env, "EMPTY", {}, g_args), None

    if "ARM_ACTION" in env.action_space.spaces:
        arm_action_space = env.action_space.spaces["ARM_ACTION"].spaces[
            "arm_action"
        ]
        arm_ctrlr = env.task.actions["ARM_ACTION"].arm_ctrlr
        base_action = None
    else:
        arm_action_space = np.zeros(7)
        arm_ctrlr = None
        base_action = [0, 0]

    if arm_action is None:
        arm_action = np.zeros(arm_action_space.shape[0])
        given_arm_action = False
    else:
        given_arm_action = True

    end_ep = False
    magic_grasp = None

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        return None, None
    elif keys[pygame.K_m]:
        end_ep = True

    # Base control
    elif keys[pygame.K_j]:
        # Left
        base_action = [0, 1]
    elif keys[pygame.K_l]:
        # Right
        base_action = [0, -1]
    elif keys[pygame.K_k]:
        # Back
        base_action = [-1, 0]
    elif keys[pygame.K_i]:
        # Forward
        base_action = [1, 0]

    if arm_action_space.shape[0] == 7:
        # Velocity control. A different key for each joint
        if keys[pygame.K_q]:
            arm_action[0] = 1.0
        elif keys[pygame.K_1]:
            arm_action[0] = -1.0

        elif keys[pygame.K_w]:
            arm_action[1] = 1.0
        elif keys[pygame.K_2]:
            arm_action[1] = -1.0

        elif keys[pygame.K_e]:
            arm_action[2] = 1.0
        elif keys[pygame.K_3]:
            arm_action[2] = -1.0

        elif keys[pygame.K_r]:
            arm_action[3] = 1.0
        elif keys[pygame.K_4]:
            arm_action[3] = -1.0

        elif keys[pygame.K_t]:
            arm_action[4] = 1.0
        elif keys[pygame.K_5]:
            arm_action[4] = -1.0

        elif keys[pygame.K_y]:
            arm_action[5] = 1.0
        elif keys[pygame.K_6]:
            arm_action[5] = -1.0

        elif keys[pygame.K_u]:
            arm_action[6] = 1.0
        elif keys[pygame.K_7]:
            arm_action[6] = -1.0
    elif isinstance(arm_ctrlr, ArmEEAction):
        EE_FACTOR = 0.5
        # End effector control
        if keys[pygame.K_d]:
            arm_action[1] -= EE_FACTOR
        elif keys[pygame.K_a]:
            arm_action[1] += EE_FACTOR
        elif keys[pygame.K_w]:
            arm_action[0] += EE_FACTOR
        elif keys[pygame.K_s]:
            arm_action[0] -= EE_FACTOR
        elif keys[pygame.K_q]:
            arm_action[2] += EE_FACTOR
        elif keys[pygame.K_e]:
            arm_action[2] -= EE_FACTOR
    else:
        raise ValueError("Unrecognized arm action space")

    if keys[pygame.K_p]:
        print("[play.py]: Unsnapping")
        # Unsnap
        magic_grasp = -1
    elif keys[pygame.K_o]:
        # Snap
        print("[play.py]: Snapping")
        magic_grasp = 1

    elif keys[pygame.K_PERIOD]:
        # Print the current position of the robot, useful for debugging.
        pos = ["%.3f" % x for x in env._sim.robot.sim_obj.translation]
        print(pos)

    args = {}
    if base_action is not None and "BASE_VELOCITY" in env.action_space.spaces:
        name = "BASE_VELOCITY"
        args = {"base_vel": base_action}
    else:
        name = "ARM_ACTION"
        if given_arm_action:
            # The grip is also contained in the provided action
            args = {
                "arm_action": arm_action[:-1],
                "grip_action": arm_action[-1],
            }
        else:
            args = {"arm_action": arm_action, "grip_action": magic_grasp}

    if end_ep:
        env.reset()

    if magic_grasp is None:
        arm_action = [*arm_action, 0.0]
    else:
        arm_action = [*arm_action, magic_grasp]

    return step_env(env, name, args, g_args), arm_action


def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)

    return None


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)
            print("Loaded arm actions")

    obs = env.reset()

    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        pygame.init()
        screen = pygame.display.set_mode(
            [draw_obs.shape[1], draw_obs.shape[0]]
        )

    i = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0
    all_arm_actions = []

    while True:
        if render_steps_limit is not None and i > render_steps_limit:
            break
        step_result, arm_action = get_input_vel_ctlr(
            args.no_render,
            use_arm_actions[i] if use_arm_actions is not None else None,
            args,
            obs,
            env,
        )
        if step_result is None:
            break
        all_arm_actions.append(arm_action)
        i += 1
        if use_arm_actions is not None and i >= len(use_arm_actions):
            break

        obs = step_result
        info = env.get_metrics()
        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        use_ob = observations_to_image(obs, info)
        use_ob = overlay_frame(use_ob, info)

        draw_ob = use_ob[:]

        if not args.no_render:
            draw_ob = np.transpose(draw_ob, (1, 0, 2))
            draw_obuse_ob = pygame.surfarray.make_surface(draw_ob)
            screen.blit(draw_obuse_ob, (0, 0))
            pygame.display.update()
        if args.save_obs:
            all_obs.append(draw_ob)

        if not args.no_render:
            pygame.event.pump()
        if env.episode_over:
            total_reward = 0
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_actions:
        if len(all_arm_actions) < args.save_actions_count:
            raise ValueError(
                f"Only did {len(all_arm_actions)} actions but {args.save_actions_count} are required"
            )
        all_arm_actions = np.array(all_arm_actions)[: args.save_actions_count]
        os.makedirs(SAVE_ACTIONS_DIR, exist_ok=True)
        save_path = osp.join(SAVE_ACTIONS_DIR, args.save_actions_fname)
        with open(save_path, "wb") as f:
            np.save(f, all_arm_actions)
        print(f"Saved actions to {save_path}")
        pygame.quit()
        return

    if args.save_obs:
        all_obs = np.array(all_obs)
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(all_obs, osp.join(SAVE_VIDEO_DIR, args.save_obs_fname))
    if not args.no_render:
        pygame.quit()


def has_pygame():
    return pygame is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--play-task",
        action="store_true",
        default=False,
        help="If true, then change the config settings to make it easier to play and visualize the task.",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--add-ik",
        action="store_true",
        default=False,
        help="If true, changes arm control to IK",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    if not has_pygame() and not args.no_render:
        raise ImportError(
            "Need to install PyGame (run `pip install pygame==2.0.1`)"
        )

    config = habitat.get_config(args.cfg, args.opts)
    config.defrost()
    if args.play_task:
        config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH = args.play_cam_res
        config.SIMULATOR.THIRD_RGB_SENSOR.HEIGHT = args.play_cam_res
        config.SIMULATOR.AGENT_0.SENSORS.append("THIRD_RGB_SENSOR")
    if args.never_end:
        config.ENVIRONMENT.MAX_EPISODE_STEPS = 0
    if args.add_ik:
        config.TASK.ACTIONS.ARM_ACTION.ARM_CONTROLLER = "ArmEEAction"
        config.SIMULATOR.IK_ARM_URDF = (
            "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
        )
    config.freeze()

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
