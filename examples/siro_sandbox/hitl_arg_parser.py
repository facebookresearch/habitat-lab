#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Final

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH: Final[str] = (
    # TODO: Get from model.
    "data/humanoids/humanoid_data/walking_motion_processed_smplx.pkl"
)

DEFAULT_CFG: Final[
    str
] = "experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml"


def create_hitl_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-sps",
        type=int,
        default=30,
        help="Target rate to step the environment (steps per second); actual SPS may be lower depending on your hardware",
    )
    parser.add_argument(
        "--width",
        default=1280,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=720,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--gui-controlled-agent-index",
        type=int,
        default=None,
        help=(
            "GUI-controlled agent index (must be >= 0 and < number of agents). "
            "Defaults to None, indicating that all the agents are policy-controlled. "
            "If none of the agents is GUI-controlled, the camera is switched to 'free camera' mode "
            "that lets the user observe the scene (instead of controlling one of the agents)"
        ),
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control. Only relevant for a user-controlled *robot* agent.",
    )
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "--cfg-opts",
        nargs="*",
        default=list(),
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--debug-images",
        nargs="*",
        default=list(),
        help=(
            "Visualize camera sensors (corresponding to `--debug-images` keys) in the app GUI."
            "For example, to visualize agent1's head depth sensor set: --debug-images agent_1_head_depth"
        ),
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )
    parser.add_argument(
        "--lin-speed",
        type=float,
        default=10.0,
        help="GUI-controlled agent's linear speed",
    )
    parser.add_argument(
        "--ang-speed",
        type=float,
        default=10.0,
        help="GUI-controlled agent's angular speed",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--use-batch-renderer",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )
    parser.add_argument(
        "--debug-third-person-width",
        default=0,
        type=int,
        help="If specified, enable the debug third-person camera (habitat.simulator.debug_render) with specified viewport width",
    )
    parser.add_argument(
        "--debug-third-person-height",
        default=0,
        type=int,
        help="If specified, use the specified viewport height for the debug third-person camera",
    )
    parser.add_argument(
        "--max-look-up-angle",
        default=15,
        type=float,
        help="Look up angle limit.",
    )
    parser.add_argument(
        "--min-look-down-angle",
        default=-60,
        type=float,
        help="Look down angle limit.",
    )
    parser.add_argument(
        "--first-person-mode",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )
    parser.add_argument(
        "--can-grasp-place-threshold",
        default=1.2,
        type=float,
        help="Object grasp/place proximity threshold",
    )
    parser.add_argument(
        "--episodes-filter",
        default=None,
        type=str,
        help=(
            "Episodes filter in the form '0:10 12 14:20:2', "
            "where single integer number (`12` in this case) represents an episode id, "
            "colon separated integers (`0:10' and `14:20:2`) represent start:stop:step ids range."
        ),
    )
    parser.add_argument(
        "--hide-humanoid-in-gui",
        action="store_true",
        default=False,
        help="Hide the humanoid in the GUI viewport. Note it will still be rendered into observations fed to policies. This option is a workaround for broken skinned humanoid rendering in the GUI viewport.",
    )
    parser.add_argument(
        "--save-gfx-replay-keyframes",
        action="store_true",
        default=False,
        help="Save the gfx-replay keyframes to file. Use --save-filepath-base to specify the filepath base.",
    )
    parser.add_argument(
        "--save-episode-record",
        action="store_true",
        default=False,
        help="Save recorded episode data to file. Use --save-filepath-base to specify the filepath base.",
    )
    parser.add_argument(
        "--save-filepath-base",
        default=None,
        type=str,
        help="Filepath base used for saving various session data files. Include a full path including basename, but not an extension.",
    )
    parser.add_argument(
        "--app-state",
        default="rearrange",
        type=str,
        help="'rearrange', 'pick_throw_vr', 'socialnav' or 'free_camera'",
    )
    parser.add_argument(
        "--remote-gui-mode",
        action="store_true",
        default=False,
        help="When enabled, the sandbox app behaves as a server that takes input from a remote client.",
    )
    return parser


def get_hitl_args():
    args = create_hitl_arg_parser().parse_args()
    if (
        args.save_gfx_replay_keyframes or args.save_episode_record
    ) and not args.save_filepath_base:
        raise ValueError(
            "--save-gfx-replay-keyframes and/or --save-episode-record flags are enabled, "
            "but --save-filepath-base argument is not set. Specify filepath base for the session episode data to be saved."
        )

    return args
