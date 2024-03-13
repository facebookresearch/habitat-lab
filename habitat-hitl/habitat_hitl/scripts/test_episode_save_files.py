#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np

from habitat_hitl.core.serialize_utils import load_json_gzip, load_pickle_gzip


def get_comp_value(a):
    if isinstance(a, np.generic):
        # If obj is a NumPy scalar type, convert it to a standard Python scalar type
        return a.item()
    else:
        return a


# Not completely general-purpose. Works for simple types and list-like types.
def approx_equal(a_src, b_src, eps=1e-4):
    a = get_comp_value(a_src)
    b = get_comp_value(b_src)

    if isinstance(a, (int, str, bool, type(None))):
        return a == b
    elif isinstance(a, float):
        return abs(a - b) < eps
    else:
        a_list = a
        b_list = b
        if len(a_list) != len(b_list):
            return False

        return all(
            approx_equal(a_list[i], b_list[i]) for i in range(len(a_list))
        )


def test_episode_save_files(filepath_base):
    json_filepath = filepath_base + ".json.gz"
    pkl_filepath = filepath_base + ".pkl.gz"

    pkl_obj = load_pickle_gzip(pkl_filepath)
    json_obj = load_json_gzip(json_filepath)

    # Spot-check selected items for equality. We don't expect equality in all items
    # because of differences in how the two data files are serialized/deserialized.
    # For example, for json, some complex types are converted to strings and floats
    # are rounded to 5 digits of precision.

    for key in ["scene_id", "episode_id", "target_obj_ids"]:
        assert pkl_obj[key] == json_obj[key]

    for key in ["goal_positions"]:
        assert len(pkl_obj[key])
        assert approx_equal(pkl_obj[key], json_obj[key])

    for step_idx in range(len(pkl_obj["steps"])):
        pkl_step = pkl_obj["steps"][step_idx]
        json_step = json_obj["steps"][step_idx]

        assert approx_equal(
            pkl_step["gui_humanoid"]["cam_yaw"],
            json_step["gui_humanoid"]["cam_yaw"],
        )

        # These will not be present if the episode didn't include a policy-driven robot.
        pkl_subobj = pkl_step["action"]["action_args"]
        json_subobj = json_step["action"]["action_args"]
        keys = [
            "agent_0_arm_action",
            "agent_0_oracle_nav_with_backing_up_action",
        ]
        for key in keys:
            assert len(pkl_subobj[key])
            assert approx_equal(pkl_subobj[key], json_subobj[key])

        # We chose not to save these because it's too much data and bloats save files.
        assert pkl_subobj["agent_1_human_joints_trans"] == None
        assert json_subobj["agent_1_human_joints_trans"] == None

        pkl_agent_states = pkl_step["agent_states"]
        json_agent_states = pkl_step["agent_states"]
        assert len(pkl_agent_states)
        keys = ["position", "rotation_xyzw", "grasp_mgr_snap_idx"]
        for i in range(len(pkl_agent_states)):
            pkl_subobj = pkl_agent_states[i]
            json_subobj = json_agent_states[i]
            for key in keys:
                assert approx_equal(pkl_subobj[key], json_subobj[key])

        pkl_subobj = pkl_step["target_object_positions"]
        json_subobj = json_step["target_object_positions"]
        assert len(pkl_subobj)
        approx_equal(pkl_subobj, json_subobj)

    print("success!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-filepath-base",
        type=str,
        help="Same filepath base used for the sandbox app's --save-filepath-base.",
    )
    args = parser.parse_args()

    test_episode_save_files(args.load_filepath_base)
