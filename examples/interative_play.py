#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import os.path as osp
import time

import cv2
import numpy as np

import habitat.tasks.rearrange.rearrange_task
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.config.default import get_config

try:
    import pygame
except ImportError:
    pass

DEFAULT_CFG = "configs/tasks/rearrangepick_replica_cad_example.yaml"

def make_video_cv2(observations, prefix=""):
    output_path = "./data/vids/"
    if not osp.exists(output_path):
        os.makedirs(output_path)
    shp = observations[0].shape
    videodims = (shp[1], shp[0])
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vid_name = output_path + prefix + ".mp4"
    video = cv2.VideoWriter(vid_name, fourcc, 60, videodims)
    for ob in observations:
        bgr_im_1st_person = ob[..., 0:3][..., ::-1]
        video.write(bgr_im_1st_person)
    video.release()
    print("Saved to", vid_name)

def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]

    y = 0
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    text_image = cv2.flip(text_image, 0)
    final = np.concatenate((image, text_image), axis=0)
    return final

def overlay_frame(frame, info):
    lines = []
    if 'object_to_goal_distance' in info:
        lines.append("Obj to goal %.2f" % info['object_to_goal_distance'][0])
    if 'ee_to_object_distance' in info:
        lines.append("EE to obj %.2f" % info['ee_to_object_distance'][0])
    if 'robo_force' in info:
        lines.append("Force: %.2f" % info['robo_force'])
    if 'robo_collisions' in info:
        coll_info = info['robo_collisions']
        lines.extend([
            "Obj-Scene Coll: %.2f" % coll_info['obj_scene_colls'],
            "Robo-Obj Coll: %.2f" % coll_info['robo_obj_colls'],
            "Robo-Scene Coll: %.2f" % coll_info['robo_scene_colls'],
            ])

    frame = append_text_to_image(frame, lines)

    return frame


def step_env(env, action_name, action_args, args):
    return env.step({"action": action_name, "action_args": action_args})


def get_input_vel_ctlr(skip_pygame, arm_action, g_args, prev_obs, env):
    if skip_pygame:
        return step_env(env, "EMPTY", {}, g_args), None

    arm_action_space = env.action_space.spaces["ARM_ACTION"].spaces["arm_ac"]

    arm_action = np.zeros(arm_action_space.shape[0])
    base_action = None
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
        base_action = [0, -1]
    elif keys[pygame.K_l]:
        # Right
        base_action = [0, 1]
    elif keys[pygame.K_k]:
        # Back
        base_action = [-1, 0]
    elif keys[pygame.K_i]:
        # Forward
        base_action = [1, 0]

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
        pos = ['%.3f' % x for x in env._sim.robot.sim_obj.translation]
        print(pos)

    args = {}
    if base_action is not None:
        name = "BASE_VEL"
        args = {"base_vel": base_action}
    else:
        name = "ARM_ACTION"
        args = {"arm_ac": arm_action, "grip_ac": magic_grasp}

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
    if not args.no_render:
        pygame.init()
        render_dim = config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH
        screen = pygame.display.set_mode([render_dim, render_dim])

    render_count = None
    if args.no_render:
        render_count = 60 * 60

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)

    obs = env.reset()
    i = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0
    all_arm_actions = []

    while True:
        if render_count is not None and i > render_count:
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

        # obs, reward, done, info = step_result
        obs = step_result
        reward = 0.0
        info = env.get_metrics()

        total_reward += reward

        obs["rgb"] = obs["robot_third_rgb"]
        use_ob = observations_to_image(obs, info)
        use_ob = overlay_frame(use_ob, info)

        if len(use_ob) == 1:
            use_ob = use_ob[0]
        draw_ob = use_ob[:]

        if not args.no_render:
            draw_ob = np.flip(draw_ob, 0)
            draw_ob = np.transpose(draw_ob, (1, 0, 2))
            draw_obuse_ob = pygame.surfarray.make_surface(draw_ob)
            screen.blit(draw_obuse_ob, (0, 0))
            pygame.display.update()
        if args.save_obs:
            all_obs.append(draw_ob)

        if not args.no_render:
            pygame.event.pump()
        if env.episode_over:
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_actions:
        assert len(all_arm_actions) > 200
        all_arm_actions = np.array(all_arm_actions)[:200]
        save_dir = "orp/start_data/"
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_path = osp.join(save_dir, "bench_ac.txt")
        with open(save_path, "wb") as f:
            np.save(f, all_arm_actions)
        raise ValueError("done")

    if args.save_obs:
        all_obs = np.array(all_obs)
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))
        make_video_cv2(all_obs, "interactive_play")
    pygame.quit()


if __name__ == "__main__":
    """
    Manually control the robot to interact with the environment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    args = parser.parse_args()

    config = habitat.get_config(args.cfg)

    with habitat.Env(config=config) as env:
        play_env(env, args, config)


