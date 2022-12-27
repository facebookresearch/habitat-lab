#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the robot to interact with the environment. Run as
```
python examples/interative_play.py
```

To Run you need PyGame installed (to install run `pip install pygame==2.0.1`).

By default this controls with velocity control (which makes controlling the
robot hard). To use IK control instead add the `--add-ik` command line argument.

Controls:
- For velocity control
    - 1-7 to increase the motor target for the robot arm joints
    - Q-U to decrease the motor target for the robot arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the robot base around
- PERIOD to print the current world coordinates of the robot base.
- Z to toggle the camera to free movement mode. When in free camera mode:
    - W,S,A,D,Q,E to translate the camera
    - I,J,K,L,U,O to rotate the camera
    - B to reset the camera position
- X to change the robot that is being controlled (if there are multiple robots).

Change the task with `--cfg configs/tasks/rearrange/close_cab.yaml` (choose any task under the `configs/tasks/rearrange/` folder).

Change the grip type:
- Suction gripper `TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER "SuctionGraspAction"`

To record a video: `--save-obs` This will save the video to file under `data/vids/` specified by `--save-obs-fname` (by default `vid.mp4`).

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
from collections import defaultdict

import cv2
import magnum as mn
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_rotate_vector_v2,
)
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.utils import viz_utils as vut

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Color in BGR
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2

try:
    import pygame
except ImportError:
    pygame = None

# DEFAULT_CFG = "configs/tasks/rearrange/play_spot.yaml"
DEFAULT_CFG = "/Users/jimmytyyang/Habitat/habitat-lab/configs/tasks/rearrange/play_stretch_gripper_roll_pitch_yaw.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"

import os

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# Load the CLIP and LSEG model
import sys

sys.path.append("/Users/jimmytyyang/HomeRobot/home-robot/src")
import torch
from home_robot.agent.perception.detection.lseg import load_lseg_for_inference

checkpoint_path = "/Users/jimmytyyang/LSeg/checkpoints/demo_e200.ckpt"
device = torch.device("cpu")
model = load_lseg_for_inference(checkpoint_path, device)

from home_robot.agent.mapping.dense.semantic.vision_language_2d_semantic_map_module import (
    VisionLanguage2DSemanticMapModule,
)
from home_robot.agent.mapping.dense.semantic.vision_language_2d_semantic_map_state import (
    VisionLanguage2DSemanticMapState,
)

ENABLE_PLOT_SEM_MAP = False
MAP_SIZE_CM = 2000
TARGET_SEQ_LEN = 500
GLOAL_DOWNSCALING = 4
MAP_RESOLUTION = 4
VISION_RANGE = 50
DU_SCALE = 4
EXP_PRED_THRESHOLD = 1
OBS_PRED_THRESHOLD = 150
FOLDER_NAME = "camera_trans_v12"
TEXT_LABELS = [
    "stair",
    "tree",
    "chair",
    "couch",
    "lamp",
    "cabinet",
    "sink",
    "fridge",
    "bottle",
    "bike",
    "door",
    "other",
]
MAP_SEQ_LEN = [TARGET_SEQ_LEN]

# Make dir
SAVE_DIR = "/Users/jimmytyyang/Documents/lseg_image/" + FOLDER_NAME
try:
    os.mkdir(SAVE_DIR)
except:
    print("Save Folder Created...")
SAVE_DIR += "/"

# The robot's camera is facing +y direction, which is z in habitat
# The robot's right is facing +x direction, which is z in habitat
# The robot moves clock-wise is positive delta (cur_angle - prev_angle)
# The robot moves counter-clock-wise is negative delta (cur_angle - prev_angle)


def compute_delta_angle(base_action, cur_angle, last_angle):
    """Compute the delta angle. Habitat has a weird angle formation, we
    we need this function to get a correct delta angle."""
    # Get the control rotation
    if base_action is not None:
        rot_ctr = base_action[1]
        if rot_ctr == 1:
            # counter clock-wise
            rot_sign = -1
        elif rot_ctr == -1:
            # clock-wise
            rot_sign = 1
        else:
            return 0.0
    else:
        return 0.0

    # Get the angle delta
    decision_angle_120 = np.pi * 120.0 / 180.0
    decision_angle_240 = np.pi * 240.0 / 180.0
    # Set the threshold to detect the change between 120 and 240 degree.
    threshold = 2.0
    if abs(last_angle - cur_angle) >= threshold:
        if abs(cur_angle - decision_angle_120) < abs(
            cur_angle - decision_angle_240
        ):
            delta_angle = abs(cur_angle - decision_angle_120) + abs(
                last_angle - decision_angle_240
            )
        else:
            delta_angle = abs(cur_angle - decision_angle_240) + abs(
                last_angle - decision_angle_120
            )
    else:
        delta_angle = abs(last_angle - cur_angle)
    return delta_angle * rot_sign


def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})


def get_input_vel_ctlr(
    skip_pygame, arm_action, env, not_block_input, agent_to_control
):
    if skip_pygame:
        return step_env(env, "EMPTY", {}), None, False
    multi_agent = len(env._sim.robots_mgr) > 1

    arm_action_name = "ARM_ACTION"
    base_action_name = "BASE_VELOCITY"
    arm_key = "arm_action"
    grip_key = "grip_action"
    base_key = "base_vel"
    if multi_agent:
        agent_k = f"AGENT_{agent_to_control}"
        arm_action_name = f"{agent_k}_{arm_action_name}"
        base_action_name = f"{agent_k}_{base_action_name}"
        arm_key = f"{agent_k}_{arm_key}"
        grip_key = f"{agent_k}_{grip_key}"
        base_key = f"{agent_k}_{base_key}"

    if arm_action_name in env.action_space.spaces:
        arm_action_space = env.action_space.spaces[arm_action_name].spaces[
            arm_key
        ]
        arm_ctrlr = env.task.actions[arm_action_name].arm_ctrlr
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
        return None, None, False, None
    elif keys[pygame.K_m]:
        end_ep = True
    elif keys[pygame.K_n]:
        env._sim.navmesh_visualization = not env._sim.navmesh_visualization

    if not_block_input:
        # Base control
        if keys[pygame.K_j]:
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
        elif arm_action_space.shape[0] == 8:
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

            elif keys[pygame.K_8]:
                arm_action[7] = 1.0
            elif keys[pygame.K_9]:
                arm_action[7] = -1.0
        elif arm_action_space.shape[0] == 10:
            # Velocity control. A different key for each joint
            # arm_control, arm_joints index, name
            # 0 28: joint_arm_l0
            # 1 27: joint_arm_l1
            # 2 26: joint_arm_l2
            # 3 25: joint_arm_l3
            # 4 23: joint_lift
            # 5 31: joint_wrist_yaw
            # 6 39: joint_wrist_pitch
            # 7 40: joint_wrist_roll
            # 8 7: joint_head_pan
            # 9 8: joint_head_tilt

            if keys[
                pygame.K_q
            ]:  # joint_arm_l0, joint_arm_l1, joint_arm_l2, joint_arm_l3
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:  # joint_lift
                arm_action[4] = 1.0
            elif keys[pygame.K_2]:
                arm_action[4] = -1.0

            elif keys[pygame.K_e]:  # joint_wrist_yaw
                arm_action[5] = 1.0
            elif keys[pygame.K_3]:
                arm_action[5] = -1.0

            elif keys[pygame.K_r]:  # joint_wrist_pitch
                arm_action[6] = 1.0
            elif keys[pygame.K_4]:
                arm_action[6] = -1.0

            elif keys[pygame.K_t]:  # joint_wrist_roll
                arm_action[7] = 1.0
            elif keys[pygame.K_5]:
                arm_action[7] = -1.0

            elif keys[pygame.K_y]:  # joint_head_pan
                arm_action[8] = 1.0
            elif keys[pygame.K_6]:
                arm_action[8] = -1.0

            elif keys[pygame.K_u]:  # joint_head_tilt
                arm_action[9] = 1.0
            elif keys[pygame.K_7]:
                arm_action[9] = -1.0

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
            logger.info("[play.py]: Unsnapping")
            # Unsnap
            magic_grasp = -1
        elif keys[pygame.K_o]:
            # Snap
            logger.info("[play.py]: Snapping")
            magic_grasp = 1

    if keys[pygame.K_PERIOD]:
        # Print the current position of the robot, useful for debugging.
        pos = [float("%.3f" % x) for x in env._sim.robot.sim_obj.translation]
        rot = env._sim.robot.sim_obj.rotation
        ee_pos = env._sim.robot.ee_transform.translation
        logger.info(
            f"Robot state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        )
    elif keys[pygame.K_COMMA]:
        # Print the current arm state of the robot, useful for debugging.
        joint_state = [float("%.3f" % x) for x in env._sim.robot.arm_joint_pos]
        logger.info(f"Robot arm joint state: {joint_state}")

    args = {}
    if base_action is not None and base_action_name in env.action_space.spaces:
        name = base_action_name
        args = {base_key: base_action}
    else:
        name = arm_action_name
        if given_arm_action:
            # The grip is also contained in the provided action
            args = {
                arm_key: arm_action[:-1],
                grip_key: arm_action[-1],
            }
        else:
            args = {arm_key: arm_action, grip_key: magic_grasp}

    if magic_grasp is None:
        arm_action = [*arm_action, 0.0]
    else:
        arm_action = [*arm_action, magic_grasp]

    return step_env(env, name, args), arm_action, end_ep, base_action


def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)

    return None


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


class FreeCamHelper:
    def __init__(self):
        self._is_free_cam_mode = False
        self._last_pressed = 0
        self._free_rpy = np.zeros(3)
        self._free_xyz = np.zeros(3)

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def update(self, env, step_result, update_idx):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_z] and (update_idx - self._last_pressed) > 60:
            self._is_free_cam_mode = not self._is_free_cam_mode
            logger.info(f"Switching camera mode to {self._is_free_cam_mode}")
            self._last_pressed = update_idx

        if self._is_free_cam_mode:
            offset_rpy = np.zeros(3)
            if keys[pygame.K_u]:
                offset_rpy[1] += 1
            elif keys[pygame.K_o]:
                offset_rpy[1] -= 1
            elif keys[pygame.K_i]:
                offset_rpy[2] += 1
            elif keys[pygame.K_k]:
                offset_rpy[2] -= 1
            elif keys[pygame.K_j]:
                offset_rpy[0] += 1
            elif keys[pygame.K_l]:
                offset_rpy[0] -= 1

            offset_xyz = np.zeros(3)
            if keys[pygame.K_q]:
                offset_xyz[1] += 1
            elif keys[pygame.K_e]:
                offset_xyz[1] -= 1
            elif keys[pygame.K_w]:
                offset_xyz[2] += 1
            elif keys[pygame.K_s]:
                offset_xyz[2] -= 1
            elif keys[pygame.K_a]:
                offset_xyz[0] += 1
            elif keys[pygame.K_d]:
                offset_xyz[0] -= 1
            offset_rpy *= 0.1
            offset_xyz *= 0.1
            self._free_rpy += offset_rpy
            self._free_xyz += offset_xyz
            if keys[pygame.K_b]:
                self._free_rpy = np.zeros(3)
                self._free_xyz = np.zeros(3)

            quat = euler_to_quat(self._free_rpy)
            trans = mn.Matrix4.from_(
                quat.to_matrix(), mn.Vector3(*self._free_xyz)
            )
            env._sim._sensors[
                "robot_third_rgb"
            ]._sensor_object.node.transformation = trans
            step_result = env._sim.get_sensor_observations()
            return step_result
        return step_result


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)
            logger.info("Loaded arm actions")

    obs = env.reset()

    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        pygame.init()
        screen = pygame.display.set_mode(
            [draw_obs.shape[1], draw_obs.shape[0]]
        )

    update_idx = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0
    all_arm_actions = []
    agent_to_control = 0

    free_cam = FreeCamHelper()
    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )
    is_multi_agent = len(env._sim.robots_mgr) > 1

    image_i = 0

    seq_obs_all = []
    seq_pose_delta_all = []
    last_seq_pose_for_trans = None

    while True:
        if (
            args.save_actions
            and len(all_arm_actions) > args.save_actions_count
        ):
            # quit the application when the action recording queue is full
            break
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        if args.no_render:
            keys = defaultdict(lambda: False)
        else:
            keys = pygame.key.get_pressed()

        if not args.no_render and is_multi_agent and keys[pygame.K_x]:
            agent_to_control += 1
            agent_to_control = agent_to_control % len(env._sim.robots_mgr)
            logger.info(
                f"Controlled agent changed. Controlling agent {agent_to_control}."
            )

        step_result, arm_action, end_ep, base_action = get_input_vel_ctlr(
            args.no_render,
            use_arm_actions[update_idx]
            if use_arm_actions is not None
            else None,
            env,
            not free_cam.is_free_cam_mode,
            agent_to_control,
        )

        # Collect the data for semantic map
        # This avoid the issue of getting the sematic map
        if step_result is None:
            break

        # Get rgb map
        rgb_for_map = np.expand_dims(step_result["robot_head_rgb"], axis=0)
        # Get the depth map. Note that the output from Habitat is in meter, it needs
        # to be cm
        depth_for_map = (
            np.expand_dims(step_result["robot_head_depth"], axis=0) * 100
        )
        # Combine rgb and depth
        obs_for_map = np.concatenate((rgb_for_map, depth_for_map), axis=3)

        # For the dataset
        if len(seq_obs_all) == 0:
            seq_obs_all.append(obs_for_map.copy())
            seq_pose_delta_temp = np.array([0.0, 0.0, 0.0])
            seq_pose_delta_all.append(
                np.expand_dims(seq_pose_delta_temp, axis=0).copy()
            )
        else:
            # Append the image observation
            seq_obs_all.append(obs_for_map.copy())

            # Get the current camera pose
            cur_camera_position = (
                env.sim.agents[0]
                .get_state()
                .sensor_states["robot_head_rgb"]
                .position.copy()
            )
            cur_camera_rotation = (
                env.sim.agents[0]
                .get_state()
                .sensor_states["robot_head_rgb"]
                .rotation.copy()
            )

            # Process the pose delta
            angle_correction = compute_delta_angle(
                base_action,
                cur_camera_rotation.angle(),
                prev_camera_rotation.angle(),
            )

            # Compute the detla of the pose
            rotation_delta = (
                prev_camera_rotation.inverse() * cur_camera_rotation
            )
            position_delta = quaternion_rotate_vector(
                prev_camera_rotation.inverse(),
                cur_camera_position,
            ) - quaternion_rotate_vector(
                prev_camera_rotation.inverse(),
                prev_camera_position,
            )
            # The issue of 120-240 jump
            if base_action != None:
                # Process the data
                if (
                    abs(abs(rotation_delta.angle()) - abs(angle_correction))
                    > 1e-1
                ):
                    # If there is too much differecne
                    seq_pose_delta_temp = np.array(
                        [
                            position_delta[0],
                            -position_delta[2],
                            angle_correction,
                        ]
                    )
                else:
                    seq_pose_delta_temp = np.array(
                        [
                            position_delta[0],
                            -position_delta[2],
                            np.sign(angle_correction) * rotation_delta.angle(),
                        ]
                    )
            # The issue of arm control of camera does not have the sign
            elif arm_action[8] != 0:
                seq_pose_delta_temp = np.array(
                    [
                        position_delta[0],
                        -position_delta[2],
                        -arm_action[8] * rotation_delta.angle(),
                    ]
                )
            # Normal operation mode
            else:
                seq_pose_delta_temp = np.array(
                    [
                        position_delta[0],
                        -position_delta[2],
                        np.sign(angle_correction) * rotation_delta.angle(),
                    ]
                )
            print("step:", image_i, "delta pose:", seq_pose_delta_temp)
            print("trans:", rotation_delta.angle())
            print("camera:", angle_correction)

            seq_pose_delta_all.append(
                np.expand_dims(seq_pose_delta_temp, axis=0).copy()
            )

        # Get the previous camera pose
        prev_camera_position = (
            env.sim.agents[0]
            .get_state()
            .sensor_states["robot_head_rgb"]
            .position.copy()
        )
        prev_camera_rotation = (
            env.sim.agents[0]
            .get_state()
            .sensor_states["robot_head_rgb"]
            .rotation.copy()
        )

        # Get the an rgb image
        if image_i % 100 == 0 and ENABLE_PLOT_SEM_MAP:
            rgb = torch.unsqueeze(
                torch.tensor(step_result["robot_head_rgb"]), 0
            )
            depth = torch.unsqueeze(
                torch.tensor(step_result["robot_head_depth"]), 0
            )
            # pixel_features: (batch_size, 512, H, W)
            pixel_features = model.encode(rgb)
            # Define the label
            one_hot_predictions, visualizations = model.decode(
                pixel_features, TEXT_LABELS
            )

            # Plot the RGB and visualizations.
            f, axarr = plt.subplots(1, 2, figsize=(9, 14))
            axarr[0].imshow(step_result["robot_head_rgb"])
            axarr[1].imshow(visualizations[0])
            plt.savefig(SAVE_DIR + "Lseq_frame" + str(image_i) + ".png")
            plt.close()

            # Get the text and image position
            get_text_and_img_pos = model.get_text_and_img_pos()[0]
            image_text = visualizations[0].copy()
            for key in get_text_and_img_pos:
                image_text = cv2.putText(
                    image_text,
                    key,
                    get_text_and_img_pos[key],
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            # cv2.imshow("Camera Viewer", image_text)
            cv2.imwrite(
                SAVE_DIR + "Lseq_frame" + str(image_i) + "_image_text.png",
                image_text,
            )
            # Plot the figure
            data = np.zeros(
                (
                    one_hot_predictions[0].shape[0],
                    one_hot_predictions[0].shape[1],
                )
            )
            for xx in range(one_hot_predictions[0].shape[0]):
                for yy in range(one_hot_predictions[0].shape[1]):
                    for vv in range(len(TEXT_LABELS)):
                        if one_hot_predictions[0][xx][yy][vv] == 1:
                            data[xx, yy] = vv
                            break
            values = np.unique(data.ravel())
            plt.figure(figsize=(20, 14))
            im = plt.imshow(data, interpolation="none")
            # get the colors of the values, according to the
            # colormap used by imshow
            colors = [im.cmap(im.norm(value)) for value in values]
            # create a patch (proxy artist) for every color
            patches = [
                mpatches.Patch(
                    color=colors[i],
                    label="{l}".format(l=TEXT_LABELS[int(values[i])]),
                )
                for i in range(len(values))
            ]
            # put those patched as legend-handles into the legend
            plt.legend(
                handles=patches,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.0,
                fontsize=15,
            )
            plt.savefig(
                SAVE_DIR + "Lseq_frame" + str(image_i) + "_class_text.png"
            )
            plt.close()

            # Plot 3D point cloud using Open3S
            CX_DEPTH = 239.5
            CY_DEPTH = 319.5
            FX_DEPTH = 349.20216688138674
            FY_DEPTH = 349.20216688138674
            pcd = []
            for i in range(depth.shape[1]):
                for j in range(depth.shape[2]):
                    z = depth[0, i, j, 0]
                    x = (j - CX_DEPTH) * z / FX_DEPTH
                    y = (i - CY_DEPTH) * z / FY_DEPTH
                    pcd.append([x, y, z])
            import open3d as o3d

            pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
            pcd_o3d.points = o3d.utility.Vector3dVector(
                pcd
            )  # set pcd_np as the point cloud points
            # Visualize:
            o3d.visualization.draw_geometries([pcd_o3d])

        image_i += 1

        # if base_action is not None:
        #     if base_action[1] == 1:
        #         import pdb; pdb.set_trace()

        if not args.no_render and keys[pygame.K_c]:
            pddl_action = env.task.actions["PDDL_APPLY_ACTION"]
            logger.info("Actions:")
            actions = pddl_action._action_ordering
            for i, action in enumerate(actions):
                logger.info(f"{i}: {action}")
            entities = pddl_action._entities_list
            logger.info("Entities")
            for i, entity in enumerate(entities):
                logger.info(f"{i}: {entity}")
            action_sel = input("Enter Action Selection: ")
            entity_sel = input("Enter Entity Selection: ")
            action_sel = int(action_sel)
            entity_sel = [int(x) + 1 for x in entity_sel.split(",")]
            ac = np.zeros(pddl_action.action_space["pddl_action"].shape[0])
            ac_start = pddl_action.get_pddl_action_start(action_sel)
            ac[ac_start : ac_start + len(entity_sel)] = entity_sel

            step_env(env, "PDDL_APPLY_ACTION", {"pddl_action": ac})

        if not args.no_render and keys[pygame.K_g]:
            pred_list = env.task.sensor_suite.sensors[
                "all_predicates"
            ]._predicates_list
            pred_values = step_result["all_predicates"]
            logger.info("\nPredicate Truth Values:")
            for i, (pred, pred_value) in enumerate(
                zip(pred_list, pred_values)
            ):
                logger.info(f"{i}: {pred.compact_str} = {pred_value}")

        if step_result is None:
            break

        if end_ep:
            total_reward = 0
            # Clear the saved keyframes.
            if gfx_measure is not None:
                gfx_measure.get_metric(force_get=True)
            env.reset()

        if not args.no_render:
            step_result = free_cam.update(env, step_result, update_idx)

        all_arm_actions.append(arm_action)
        update_idx += 1
        if use_arm_actions is not None and update_idx >= len(use_arm_actions):
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

        if free_cam.is_free_cam_mode:
            cam = obs["robot_third_rgb"]
            use_ob = np.zeros(draw_obs.shape)
            use_ob[:, : cam.shape[1]] = cam[:, :, :3]

        else:
            use_ob = observations_to_image(obs, info)
            if not args.skip_render_text:
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

    # Change it the numpy from list
    seq_obs_all = np.stack(seq_obs_all, axis=0)
    seq_pose_delta_all = np.stack(seq_pose_delta_all, axis=0)
    seq_obs_all = np.squeeze(seq_obs_all)
    seq_pose_delta_all = np.squeeze(seq_pose_delta_all)

    # Build the map here
    # Remove the reductant position.
    frame_i = 0
    keep_i_list = [0]
    while frame_i < len(seq_pose_delta_all):
        pose_delta = seq_pose_delta_all[frame_i]
        if np.linalg.norm(pose_delta) > 1e-6:
            keep_i_list.append(frame_i)
        frame_i += 1

    # Filter out the reductant part
    seq_obs_all_final = seq_obs_all[keep_i_list]
    seq_pose_delta_all_final = seq_pose_delta_all[keep_i_list]

    print(seq_pose_delta_all_final)
    print("seq length:", seq_pose_delta_all_final.shape)
    print("keep_i_list:", keep_i_list)

    for tt in MAP_SEQ_LEN:
        # Shorten the observations.
        seq_obs_all = seq_obs_all_final[0:tt].copy()
        seq_pose_delta_all = seq_pose_delta_all_final[0:tt].copy()

        # Reshape the data
        seq_obs_all = np.transpose(seq_obs_all, (0, 3, 1, 2))
        # Format the data
        seq_obs = (
            torch.from_numpy(seq_obs_all[:, :4, :, :]).unsqueeze(0).to(device)
        )
        seq_pose_delta = (
            torch.from_numpy(seq_pose_delta_all[:]).unsqueeze(0).to(device)
        )
        seq_dones = (
            torch.tensor([False] * seq_obs.shape[1]).unsqueeze(0).to(device)
        )
        seq_update_global = (
            torch.tensor([True] * seq_obs.shape[1]).unsqueeze(0).to(device)
        )

        # State holds global and local map and sensor pose
        # See class definition for argument info
        semantic_map = VisionLanguage2DSemanticMapState(
            device=device,
            num_environments=1,
            lseg_features_dim=512,
            map_resolution=MAP_RESOLUTION,
            map_size_cm=MAP_SIZE_CM,
            global_downscaling=GLOAL_DOWNSCALING,
        )
        semantic_map.init_map_and_pose()

        # Module is responsible for updating the local and global maps and poses
        # See class definition for argument info
        semantic_map_module = VisionLanguage2DSemanticMapModule(
            lseg_checkpoint_path=checkpoint_path,
            lseg_features_dim=512,
            frame_height=640,
            frame_width=480,
            camera_height=1.85,  # camera sensor height (in metres)
            hfov=56.0,  # horizontal field of view (in degrees)
            map_size_cm=MAP_SIZE_CM,  # global map size (in centimetres)
            map_resolution=MAP_RESOLUTION,  #  size of map bins (in centimeters)
            vision_range=VISION_RANGE,  # diameter of the circular region of the local map
            # that is visible by the agent located in its center (unit is
            # the number of local map cells)
            global_downscaling=GLOAL_DOWNSCALING,  # ratio of global over local map
            du_scale=DU_SCALE,  #  frame downscaling before projecting to point cloud
            exp_pred_threshold=EXP_PRED_THRESHOLD,  # number of depth points to be in bin to consider it as explored
            map_pred_threshold=OBS_PRED_THRESHOLD,  # number of depth points to be in bin to consider it as obstacle
        ).to(device)

        (
            seq_map_features,
            semantic_map.local_map,
            semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            semantic_map.local_map,
            semantic_map.global_map,
            semantic_map.local_pose,
            semantic_map.global_pose,
            semantic_map.lmb,
            semantic_map.origins,
        )

        semantic_map.local_pose = seq_local_pose[:, -1]
        semantic_map.global_pose = seq_global_pose[:, -1]
        semantic_map.lmb = seq_lmb[:, -1]
        semantic_map.origins = seq_origins[:, -1]

        print("Finished Map...")

        # Global semantic map of shape
        # (batch_size, num_channels, M, M)
        # where num_channels = 4 + 512
        # 0: obstacle map
        # 1: explored area
        # 2: current agent location
        # 3: past agent locations
        # 4: number of cell updates
        # 5, 6, .., 5 + 512: CLIP map cell features
        from home_robot.agent.perception.detection.coco_maskrcnn.coco_categories import (
            coco_categories,
            coco_categories_color_palette,
            text_label_color_palette,
        )
        from PIL import Image

        map_color_palette = [
            1.0,
            1.0,
            1.0,  # empty space
            0.6,
            0.6,
            0.6,  # obstacles
            0.95,
            0.95,
            0.95,  # explored area
            0.96,
            0.36,
            0.26,  # visited area
            # *coco_categories_color_palette,
            *text_label_color_palette,
        ]
        map_color_palette = [int(x * 255.0) for x in map_color_palette]
        num_sem_categories = len(TEXT_LABELS)  # len(coco_categories)

        semantic_categories_map = semantic_map.get_semantic_map(
            0,
            semantic_map_module.lseg,
            # labels=list(coco_categories.keys())[:-1] + ["other"]
            labels=TEXT_LABELS,
        )

        semantic_categories_global_map = semantic_map.get_semantic_global_map(
            0,
            semantic_map_module.lseg,
            # labels=list(coco_categories.keys())[:-1] + ["other"]
            labels=TEXT_LABELS,
        )

        # Locate the position of the text and class
        label_x = {}
        label_y = {}
        for i in range(semantic_categories_map.shape[0]):
            for j in range(semantic_categories_map.shape[1]):
                if semantic_categories_map[i][j] != len(TEXT_LABELS) - 1:
                    text = TEXT_LABELS[semantic_categories_map[i][j]]
                    if text not in label_x:
                        label_x[text] = []
                        label_y[text] = []
                    label_x[text].append(i)
                    label_y[text].append(j)

        # Plot the map
        plt.figure(figsize=(20, 14))
        for text in label_x:
            plt.scatter(label_x[text], label_y[text], label=text)
        plt.legend()
        plt.savefig(SAVE_DIR + "Sem_Map_Raw_" + str(tt) + ".png")
        plt.close()

        obstacle_map = semantic_map.get_obstacle_map(0)
        explored_map = semantic_map.get_explored_map(0)
        visited_map = semantic_map.get_visited_map(0)

        semantic_categories_map += 4
        # "other" class
        no_category_mask = (
            semantic_categories_map == 4 + num_sem_categories - 1
        )
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = visited_map == 1
        semantic_categories_map[no_category_mask] = 0
        semantic_categories_map[
            np.logical_and(no_category_mask, explored_mask)
        ] = 2
        # If we cannot identify there is a class there, we just treat it as a obstacle
        semantic_categories_map[
            np.logical_and(no_category_mask, obstacle_mask)
        ] = 1
        semantic_categories_map[visited_mask] = 3

        semantic_map_vis = Image.new("P", semantic_categories_map.shape)
        semantic_map_vis.putpalette(map_color_palette)
        semantic_map_vis.putdata(
            semantic_categories_map.flatten().astype(np.uint8)
        )
        semantic_map_vis = semantic_map_vis.convert("RGB")
        # Change it to array.
        semantic_map_vis_flip = np.flipud(semantic_map_vis)
        semantic_map_vis_unflip = np.flipud(semantic_map_vis_flip)
        plt.imshow(semantic_map_vis_flip)
        plt.savefig(SAVE_DIR + "Sem_Map_Package_" + str(tt) + ".png")
        plt.close()

        plt.imshow(semantic_map_vis_unflip)
        plt.savefig(SAVE_DIR + "Unflip_Sem_Map_Package_" + str(tt) + ".png")
        plt.close()

        # Add text
        semantic_map_vis_text = np.array(semantic_map_vis.copy())
        for key in label_x:
            x_pos = label_y[key][0]
            y_pos = label_x[key][0]
            semantic_map_vis_text = cv2.putText(
                semantic_map_vis_text,
                key,
                (x_pos, y_pos),
                font,
                fontScale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(
            SAVE_DIR + "Sem_Map_Package_Text_" + str(tt) + ".png",
            semantic_map_vis_text,
        )

        print("Finished Local Map...")

        # Get the same thing for the global map
        obstacle_map = semantic_map.get_obstacle_global_map(0)
        explored_map = semantic_map.get_explored_global_map(0)
        visited_map = semantic_map.get_visited_global_map(0)

        semantic_categories_global_map += 4
        no_category_mask = (
            semantic_categories_global_map == 4 + num_sem_categories - 1
        )
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = visited_map == 1
        semantic_categories_global_map[no_category_mask] = 0
        semantic_categories_global_map[
            np.logical_and(no_category_mask, explored_mask)
        ] = 2
        semantic_categories_global_map[
            np.logical_and(no_category_mask, obstacle_mask)
        ] = 1
        semantic_categories_global_map[visited_mask] = 3

        semantic_map_vis = Image.new("P", semantic_categories_global_map.shape)
        semantic_map_vis.putpalette(map_color_palette)
        semantic_map_vis.putdata(
            semantic_categories_global_map.flatten().astype(np.uint8)
        )
        semantic_map_vis = semantic_map_vis.convert("RGB")
        # Change it to array.
        semantic_map_vis_flip = np.flipud(semantic_map_vis)
        plt.imshow(semantic_map_vis_flip)
        plt.savefig(SAVE_DIR + "Sem_Global_Map_Package_" + str(tt) + ".png")
        plt.close()

        semantic_map_vis_flip = np.flipud(semantic_map_vis)
        semantic_map_vis_unflip = np.flipud(semantic_map_vis_flip)
        plt.imshow(semantic_map_vis_unflip)
        plt.savefig(
            SAVE_DIR + "Unflip_Sem_Global_Map_Package_" + str(tt) + ".png"
        )
        plt.close()

        # Add text
        semantic_map_vis_text = np.array(semantic_map_vis.copy())
        for key in label_x:
            x_pos = label_y[key][0]
            y_pos = label_x[key][0]
            semantic_map_vis_text = cv2.putText(
                semantic_map_vis_text,
                key,
                (x_pos, y_pos),
                font,
                fontScale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(
            SAVE_DIR + "Sem_Global_Map_Package_Text_" + str(tt) + ".png",
            semantic_map_vis_text,
        )

        # Save the legend
        colors = []
        texts = []
        text_i = 0
        for cc in range(0, len(text_label_color_palette), 3):
            r = text_label_color_palette[cc]
            g = text_label_color_palette[cc + 1]
            b = text_label_color_palette[cc + 2]
            temp = (r, g, b)
            colors.append(temp)
            texts.append(TEXT_LABELS[text_i])
            text_i += 1
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("s", colors[i]) for i in range(text_i)]
        labels = texts
        legend = plt.legend(
            handles, labels, loc=3, framealpha=1, frameon=False
        )
        export_legend(legend, SAVE_DIR + "Legend_" + str(tt) + ".png")

        print("Finished Global Map...")

    import pdb

    pdb.set_trace()

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
        logger.info(f"Saved actions to {save_path}")
        pygame.quit()
        return

    if args.save_obs:
        all_obs = np.array(all_obs)
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )
    if gfx_measure is not None:
        gfx_str = gfx_measure.get_metric(force_get=True)
        write_gfx_replay(gfx_str, config.TASK, env.current_episode.episode_id)

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
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
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
    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
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
    if not args.same_task:
        config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH = args.play_cam_res
        config.SIMULATOR.THIRD_RGB_SENSOR.HEIGHT = args.play_cam_res
        config.SIMULATOR.AGENT_0.SENSORS.append("THIRD_RGB_SENSOR")
        config.SIMULATOR.DEBUG_RENDER = True
        config.TASK.COMPOSITE_SUCCESS.MUST_CALL_STOP = False
        config.TASK.REARRANGE_NAV_TO_OBJ_SUCCESS.MUST_CALL_STOP = False
        config.TASK.FORCE_TERMINATE.MAX_ACCUM_FORCE = -1.0
        config.TASK.FORCE_TERMINATE.MAX_INSTANT_FORCE = -1.0
    if args.gfx:
        config.SIMULATOR.HABITAT_SIM_V0.ENABLE_GFX_REPLAY_SAVE = True
        config.TASK.MEASUREMENTS.append("GFX_REPLAY_MEASURE")
    if args.never_end:
        config.ENVIRONMENT.MAX_EPISODE_STEPS = 0
    if args.add_ik:
        if "ARM_ACTION" not in config.TASK.ACTIONS:
            raise ValueError(
                "Action space does not have any arm control so incompatible with `--add-ik` option"
            )
        config.TASK.ACTIONS.ARM_ACTION.ARM_CONTROLLER = "ArmEEAction"
        config.SIMULATOR.IK_ARM_URDF = (
            "data/robots/hab_spot_arm/urdf/hab_spot_onlyarm.urdf"
        )
    config.freeze()

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
