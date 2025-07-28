#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import datetime
import gzip
import json
import math
import os
import pprint
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import magnum as mn
import numpy as np
from tqdm import tqdm

from scripts.frame_recorder import FrameEvent

# NOTE: use known version days to split sessions by expected features or contents
# NOTE: pilot is v0.0.1 and v0.0.2
# NOTE: pre-versioning, post-pilot is v0.0.3 (no verification of content at this point)
version_days = {
    "0.0.1": (datetime.date(2025, 6, 10), "task prompts and bug fixes"),
    "0.0.2": (
        datetime.date(2025, 6, 17),
        "collision detection and IK clamping",
    ),
    "0.0.3": (
        datetime.date(2025, 7, 17),
        "after post-pilot feature pause collection resumes, before explicit versioning.",
    ),
    "0.1.0": (
        datetime.date(2025, 7, 24),
        "start of explicit versioning, from here code is versioned.",
    ),
}


def compare_version(version: str, compare_to_version: str) -> int:
    """
    Compares two version strings and returns whether the first is newer(1), same(0) or older(-1).
    Assumes semantic versioning: https://semver.org/
    I.e. "<major>.<minor>.<patch>"
    """
    for v_level in range(3):
        ver = version.split(".")[v_level]
        compare_to_ver = compare_to_version.split(".")[v_level]
        if ver > compare_to_ver:
            # first version is newer at this level
            return 1
        elif ver == compare_to_ver:
            # this level is the same
            continue
        else:
            # second version is newer at this level
            return -1
    # all equal, so same version
    return 0


# format of the strings changed mid-pilot, so we'll need to merge data
rollout_dir_format = "timestamp_first"
# rollout_dir_format = "timestamp_last"


# this vector maps simulated robot dof indices to the expected hardware dof order
# NOTE: reformats the pose for hardware deployment.
murp_full_pose_converter = [
    # left_arm
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    # left hand
    26,
    34,
    42,
    50,
    28,
    36,
    44,
    52,
    29,
    37,
    45,
    53,
    27,
    35,
    43,
    51,
    # right arm
    13,
    15,
    17,
    19,
    21,
    23,
    25,
    # right hand
    30,
    38,
    46,
    54,
    32,
    40,
    48,
    56,
    33,
    41,
    49,
    57,
    31,
    39,
    47,
    55,
]


def load_json_gz(file_path: str) -> Dict[Any, Any]:
    """
    Return a Dict with the contents of the json.gz file.
    """
    with gzip.open(file_path, "rt") as f:
        return json.load(f)


def save_json_gz(json_data: Dict[str, Any], filename: str) -> None:
    """
    Save a Dict into a .json.gz file.
    """
    assert filename.endswith(".json.gz")
    with gzip.open(filename, "wt", encoding="utf-8") as f:
        json.dump(json_data, f)


def convert_timestamp(timestamp: int) -> datetime.datetime:
    """
    Converts the timestamp in seconds to a date | time
    """
    date_time = datetime.datetime.fromtimestamp(timestamp)
    return date_time


def get_episodes_from_session(session_dir: str) -> List[str]:
    """
    Scrapes a session directory to collect the episodes.
    """
    ep_records = [
        name
        for name in os.listdir(session_dir)
        if name.endswith(".json.gz") and "session" not in name
    ]
    # pprint(ep_records)
    return ep_records


def get_ep_success(ep_json: Dict[Any, Any]) -> bool:
    """
    Return a binary interpretation of the task_percent_complete flag from within an episode record's metadata.
    """
    if ep_json["episode"]["finished"] == True:
        succeeded = float(ep_json["episode"]["task_percent_complete"]) > 0.0
        # print(f"succeeded = {succeeded}, %complete = {float(ep_json['episode']['task_percent_complete'])}")
        return succeeded
    return False


def get_event_frames(
    ep_frames_json: List[Dict[Any, Any]], e: FrameEvent
) -> List[int]:
    """
    Return all frames in the sequence which contain the specified FrameEvent in ascending order.
    """
    event_indices = []
    key_string = str(e)
    for ix, frame_data in enumerate(ep_frames_json):
        if key_string in frame_data.get("events", []):
            event_indices.append(ix)
    return event_indices


def get_good_first_ep_frame(
    ep_frames_json: List[Dict[Any, Any]], skip_reset_events: bool = True
) -> int:
    """
    From an episode frames JSON dict finds a good first frame for replaying relevant content.

    :param skip_reset_events: If True, skip any frames before a reset event tag. For example, nothing before a robot base reset is considered.
    """

    start_frame_ix = None

    # []"frames"][<ix>]
    robot_turning_threshold = 0.0001
    robot_translating_threshold = 0.001  # NOTE: only look at xz
    robot_moving_min_time = 1.0
    robot_teleport_threshold = 0.25  # NOTE: only look at xz
    # object_moving_threshold = 0.001  # NOTE: get "most moved" object

    robot_moving: Tuple[bool, float, int] = (False, None, None)
    # robot_moving = (True, <start_time>)

    # look for explicitly recorded reset events to indicate to start of an episode
    if skip_reset_events:
        max_reset_event_index = 0
        for reset_event_type in [
            FrameEvent.RESET_ARMS_FINGERS,
            FrameEvent.RESET_OBJECTS,
            FrameEvent.TELEPORT,
        ]:
            event_indices = get_event_frames(ep_frames_json, reset_event_type)
            print(f"Event {reset_event_type} at indices {event_indices}")
            if len(event_indices) > 0:
                max_reset_event_index = max(
                    max_reset_event_index, max(event_indices)
                )
        print(f"max_reset_event_index = {max_reset_event_index}")
        start_frame_ix = max_reset_event_index

    iter_start_frame = start_frame_ix if start_frame_ix is not None else 0
    prev_frame_data: Dict[Any, Any] = None
    for ix in range(iter_start_frame, len(ep_frames_json)):
        frame_data = ep_frames_json[ix]
        if prev_frame_data is not None:
            prev_base_rot_ang = (
                prev_frame_data["robot_state"]["base_rotation_angle"] + math.pi
            )
            cur_base_rot_ang = (
                frame_data["robot_state"]["base_rotation_angle"] + math.pi
            )
            delta_base_rot_ang = abs(cur_base_rot_ang - prev_base_rot_ang)
            robot_rotating = delta_base_rot_ang > robot_turning_threshold

            prev_base_pos = mn.Vector3(
                *prev_frame_data["robot_state"]["base_pos"]
            )
            cur_base_pos = mn.Vector3(*frame_data["robot_state"]["base_pos"])
            delta_base_pos = cur_base_pos - prev_base_pos
            # NOTE: remove vertical translation
            delta_base_pos[1] = 0
            robot_translating = (
                delta_base_pos.length() > robot_translating_threshold
            )

            robot_moving_now = robot_translating or robot_rotating
            if robot_moving_now and not robot_moving[0]:
                # the robot has started moving
                robot_moving = (True, frame_data["t"], ix)
            elif not robot_moving_now:
                # the robot has stopped moving
                robot_moving = (False, None, None)

            # NOTE: teleporting resets the move tracker and start_frame_ix
            robot_teleporting = (
                delta_base_pos.length() > robot_teleport_threshold
            )
            if robot_teleporting:
                # the robot has teleported
                robot_moving = (False, None, None)
                start_frame_ix = None

            if (
                (start_frame_ix is None)
                and robot_moving[0]
                and (frame_data["t"] - robot_moving[1])
                >= robot_moving_min_time
            ):
                print(
                    f"start_frame_ix = {robot_moving[2]} | set at frame {ix} with move time of {(frame_data['t']-robot_moving[1])}"
                )
                # breakpoint()
                start_frame_ix = robot_moving[2]

        prev_frame_data = frame_data

    # TODO: look for object resets
    # ["object_states"][<ix>]["object_id"] - to id the object
    # ["object_states"][<ix>]["transformation"] - reconstruct like:
    # ro_t = mn.Matrix4(
    #     [
    #         [
    #             object_state_record["transformation"][j][i]
    #             for j in range(4)
    #         ]
    #         for i in range(4)
    #     ]
    # )
    # try:
    #     ro.transformation = ro_t
    # except ValueError as e:
    #     print(
    #         f"Failed to set object transform '{ro._rigid_prim.prim_path}' with error: {e}"
    #    )

    return start_frame_ix if start_frame_ix is not None else 0


def get_robot_joint_limits_for_hardware():
    """
    Assumes the murp pose converted variable correctly maps robot configuration spaces.
    """
    lower_limit_sim = [
        -0.15999998152256012,
        -0.15999998152256012,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -3.4028234663852886e38,
        -2.743699789047241,
        -2.743699789047241,
        -1.7836999893188477,
        -1.7836999893188477,
        -2.9006998538970947,
        -2.9006998538970947,
        -3.042099952697754,
        -3.042099952697754,
        -2.80649995803833,
        -2.80649995803833,
        0.544499933719635,
        0.544499933719635,
        -3.015899658203125,
        -3.015899658203125,
        -0.4699999690055847,
        0.2629999816417694,
        -0.4699999690055847,
        -0.4699999690055847,
        -0.4699999690055847,
        0.2629999816417694,
        -0.4699999690055847,
        -0.4699999690055847,
        -0.19599997997283936,
        -0.10499999672174454,
        -0.19599997997283936,
        -0.19599997997283936,
        -0.19599997997283936,
        -0.10499999672174454,
        -0.19599997997283936,
        -0.19599997997283936,
        -0.17399999499320984,
        -0.1889999806880951,
        -0.17399999499320984,
        -0.17399999499320984,
        -0.17399999499320984,
        -0.1889999806880951,
        -0.17399999499320984,
        -0.17399999499320984,
        -0.22699998319149017,
        -0.16199998557567596,
        -0.22699998319149017,
        -0.22699998319149017,
        -0.22699998319149017,
        -0.16199998557567596,
        -0.22699998319149017,
        -0.22699998319149017,
    ]
    upper_limit_sim = [
        0.17999999225139618,
        0.17999999225139618,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        3.4028234663852886e38,
        2.743699789047241,
        2.743699789047241,
        1.7836999893188477,
        1.7836999893188477,
        2.9006998538970947,
        2.9006998538970947,
        -0.1517999917268753,
        -0.1517999917268753,
        2.80649995803833,
        2.80649995803833,
        4.516899585723877,
        4.516899585723877,
        3.015899658203125,
        3.015899658203125,
        0.4699999690055847,
        1.3960000276565552,
        0.4699999690055847,
        0.4699999690055847,
        0.4699999690055847,
        1.3960000276565552,
        0.4699999690055847,
        0.4699999690055847,
        1.6099998950958252,
        1.1629998683929443,
        1.6099998950958252,
        1.6099998950958252,
        1.6099998950958252,
        1.1629998683929443,
        1.6099998950958252,
        1.6099998950958252,
        1.7089998722076416,
        1.6439999341964722,
        1.7089998722076416,
        1.7089998722076416,
        1.7089998722076416,
        1.6439999341964722,
        1.7089998722076416,
        1.7089998722076416,
        1.6179999113082886,
        1.7189998626708984,
        1.6179999113082886,
        1.6179999113082886,
        1.6179999113082886,
        1.7189998626708984,
        1.6179999113082886,
        1.6179999113082886,
    ]

    converted_limits: List[List[float]] = [[], []]
    for ix, limit in enumerate([lower_limit_sim, upper_limit_sim]):
        converted_limits[ix] = [limit[ix] for ix in murp_full_pose_converter]

    print(converted_limits)


def get_robot_joint_poses_for_frame_range(
    frame_json: List[Dict[Any, Any]], start: int, end: int, save_to: str = None
):
    """
    Scrapes the robot's joint position vectors from the frame json and produces a numpy array of those vectors.
    Option to save the result as a .npy file.
    """

    joint_poses = []
    for frame in frame_json[start:end]:
        # Assuming joint positions are stored under "robot_state" -> "joint_positions"
        # Adjust the key as needed for your data structure
        joint_positions = frame["robot_state"]["joint_positions"]
        hardware_configuration = [
            joint_positions[ix] for ix in murp_full_pose_converter
        ]
        joint_poses.append(hardware_configuration)
    joint_poses_array = np.array(joint_poses)
    # Save to .npy file
    if save_to is not None:
        assert save_to.endswith(".npy")
        np.save(save_to, joint_poses_array)
    return joint_poses_array


def get_sessions_by_version_range(
    all_sessions: List[str], start_version: str, end_version: str = None
) -> List[str]:
    """
    Cull the list of input session directories to those which fall within the provided range.

    :param all_sessions: A list of session directory strings including the timestamp as the final postfix.
    :param start_version: The first version to allow
    :param end_version: Optionally, the last version to allow. If None, all versions after start_version are allowed.
    """
    start_version_date = version_days[start_version][0]
    end_version_date = (
        version_days[end_version][0]
        if end_version is not None
        else datetime.date.today()
    )

    matching_sessions = []
    for session_dir in all_sessions:
        if rollout_dir_format == "timestamp_first":
            session_day = convert_timestamp(
                int(session_dir.split("_")[0])
            ).date()
        elif rollout_dir_format == "timestamp_last":
            session_day = convert_timestamp(
                int(session_dir.split("_")[-1])
            ).date()
        else:
            raise ValueError()
        if session_day >= version_days["0.1.0"][0]:
            # after explicit versioning we look for the "app_version" in the episodes
            session_json = load_json_gz(
                os.path.join(session_dir, "session.json.gz")
            )
            app_version = session_json["episodes"]["0"]["app_version"]
            if (
                compare_version(app_version, start_version) >= 0
                and compare_version(app_version, end_version) <= 0
            ):
                # this session version is in-range
                matching_sessions.append(session_dir)
        elif (
            session_day >= start_version_date
            and session_day <= end_version_date
        ):
            matching_sessions.append(session_dir)
    return matching_sessions


def process_record_stats(
    data_dir: str,
    out_dir: str,
    start_version: str = None,
    end_version: str = None,
    is_local_dir: bool = False,
    do_not_process_frames: bool = False,
) -> None:
    """
    Process a directory containing teleop session records and compute statistics.

    :param data_dir: The directory containing all the session records.
    :param out_dir: The desired relative directory path where output should be saved.
    :param start_version: (optional) The first app version to include in stats calculations.
    :param end_version: (optional) The last app version to include in stats calculations.
    :param is_local_dir: If true, assume the records are cached locally from mock deployment run.
    :param do_not_process_frames: If true, don't process per-frame data to speed up the overall iteration time.
    """

    # create the output directory
    os.makedirs(out_dir, exist_ok=True)
    record_out_file = os.path.join(out_dir, "hitl_teleop_record.json")
    # record the stats to a json file
    record_json = {
        "data_dir": data_dir,
        "start_version": start_version,
        "end_version": end_version,
        "report_timestamp": str(datetime.date.today()),
        "num_completed_episodes": 0,
        "num_reported_successful_episodes": 0,
        "num_reported_failed_episodes": 0,
        "num_reported_content_bugs": 0,
        # relative path to the "data_dir"
        "reported_successful_episode_paths": [],
        "reported_failed_episode_paths": [],
        "reported_content_bug_episode_paths": [],
    }

    # collect all sessions
    session_dirs = []
    # NOTE: filtering out "test" usernames = all non-numeric usernames
    # NOTE: old style before 06/29 wiith timestamp last
    if rollout_dir_format == "timestamp_last":
        session_dirs = [
            name
            for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
            and name.split("_")[-2].isdigit()
        ]
    elif rollout_dir_format == "timestamp_first":
        # NOTE: filtering out "test" usernames = all non-numeric usernames
        # NOTE: new version with timestamp first
        session_dirs = [
            name
            for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
            and name.split("_")[-1].isdigit()
        ]
    else:
        raise ValueError()

    # NOTE: hack to adapt to local trajectory logging in mock mode
    if is_local_dir:
        session_dirs = [data_dir]
        data_dir = "."

    # print(session_dirs)

    # NOTE: Expected session directory format: <ep0>-<ep1>-...-<epn>_<user>_<timestamp>
    # cull down the sessions by version before stats processing
    if start_version is not None:
        new_session_dirs = get_sessions_by_version_range(
            session_dirs, start_version, end_version
        )
        print(
            f"culled {len(session_dirs) - len(new_session_dirs)} sessions from version range [{start_version}, {end_version}]"
        )
        session_dirs = new_session_dirs

    # TODO: eventually we'll parse by version string, but for now we'll do so by date
    session_days = defaultdict(list)
    users: defaultdict = defaultdict(int)
    if is_local_dir:
        # NOTE: hack to adapt to local trajectory logging in mock mode, so always collected today
        session_day = datetime.datetime.today().date()
    else:
        for s_dir in session_dirs:
            session_day = convert_timestamp(int(s_dir.split("_")[0])).date()
            user = s_dir.split("_")[-1]
            users[user] += 1
            session_days[session_day].append(s_dir)

    print("\nANALYSIS:\n")

    print("\nSessions by date:")
    pprint.pprint(session_days, width=25)  # Adjust the width as needed

    print("\nSessions per user:")
    pprint.pprint(users)

    # count the number of trials for each episode and sort on failures/successes
    fail_ep_paths = []
    success_ep_paths = []
    content_bug_ep_paths = []
    # dict mapping ep paths to task prompt strings, episode id, scene_id
    ep_data = {}
    ep_counts: defaultdict = defaultdict(int)

    if do_not_process_frames:
        # quick path were all data is parsed from the session metadata headers and individual episode files are not opened.
        for session_dir_name in tqdm(session_dirs):
            session_dir = os.path.join(data_dir, session_dir_name)
            session_json = load_json_gz(
                os.path.join(session_dir, "session.json.gz")
            )
            for ep_header_json in session_json["episodes"]:
                ep_id = int(ep_header_json["episode_id"])
                ep_finished = ep_header_json["finished"]
                if not ep_finished:
                    continue
                ep_counts[ep_id] += 1
                ep_path = os.path.join(session_dir, f"{ep_id}.json.gz")

                # record info from the episode json for csv export
                ep_data[ep_path] = {
                    "task_prompt": ep_header_json["episode_info"][
                        "task_prompt"
                    ],
                    "scene_id": ep_header_json["scene_id"],
                }

                content_bug = ep_header_json.get("content_bug_flagged", False)
                if content_bug:
                    content_bug_ep_paths.append(ep_path)
                success = ep_header_json["task_percent_complete"] > 0.0
                if success:
                    success_ep_paths.append(ep_path)
                else:
                    fail_ep_paths.append(ep_path)

    else:
        # get all the episodes from the sessions
        session_episodes = [
            os.path.join(data_dir, s_dir, ep)
            for s_dir in session_dirs
            for ep in get_episodes_from_session(os.path.join(data_dir, s_dir))
        ]
        pprint.pprint(session_episodes)
        frame_count_sum = 0
        # NOTE: do all the stats processing for the episodes here
        print(f"Processing {len(session_episodes)} episode records ...")
        for ep_path in tqdm(session_episodes):
            ep_index = int(ep_path.split("/")[-1].split(".json.gz")[0])

            # get the record dict from JSON
            ep_json = load_json_gz(ep_path)

            # if the episode didn't finish, don't count it
            ep_finished = ep_json["episode"]["finished"]
            if not ep_finished:
                continue

            ep_counts[ep_index] += 1

            # if a content bug was reported, log it and move on
            content_bug = ep_json["episode"].get("content_bug_flagged", False)
            if content_bug:
                content_bug_ep_paths.append(ep_path)
                continue

            good_first_frame = get_good_first_ep_frame(ep_json["frames"])
            print(f"    good first frame = {good_first_frame}")

            interesting_frames = (
                int(ep_json["episode"]["frame_count"]) - good_first_frame
            )
            frame_count_sum += interesting_frames

            # success sorting
            success = get_ep_success(ep_json)
            if success:
                success_ep_paths.append(ep_path)
            else:
                fail_ep_paths.append(ep_path)

    num_completed_episodes = len(success_ep_paths) + len(fail_ep_paths)

    # print("Episode frequency:")
    # pprint.pprint(ep_counts)
    print("Successful Episodes:")
    pprint.pprint(success_ep_paths)

    print(f"Successful: {len(success_ep_paths)}")
    print(f"Failed: {len(fail_ep_paths)}")
    print(f"Content Bugs: {len(content_bug_ep_paths)}")
    # get success vs. failure rate, but don't count content bugs
    print(f"Success rate: {len(success_ep_paths)/num_completed_episodes}")

    if not do_not_process_frames:
        avg_frames = frame_count_sum / num_completed_episodes
        print(f"Frames: total={frame_count_sum}, avg={avg_frames}")

    # record and save the stats json
    record_json["num_completed_episodes"] = num_completed_episodes
    record_json["num_reported_successful_episodes"] = len(success_ep_paths)
    record_json["num_reported_failed_episodes"] = len(fail_ep_paths)
    record_json["num_reported_content_bugs"] = len(content_bug_ep_paths)

    record_json["reported_successful_episode_paths"] = [
        os.path.relpath(ep_path, data_dir) for ep_path in success_ep_paths
    ]
    record_json["reported_failed_episode_paths"] = [
        os.path.relpath(ep_path, data_dir) for ep_path in fail_ep_paths
    ]
    record_json["reported_content_bug_episode_paths"] = [
        os.path.relpath(ep_path, data_dir) for ep_path in content_bug_ep_paths
    ]

    with open(record_out_file, "wt", encoding="utf-8") as f:
        json.dump(record_json, f)
    print(f"Wrote data to {record_out_file}")

    # write a csv file with rows for each trajectory
    csv_out_file = os.path.join(out_dir, "successful_episodes.csv")
    with open(csv_out_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["task_id"])
        for ep_path in success_ep_paths:
            dir_split_ep_path = ep_path.split("/")
            short_ep_path = dir_split_ep_path[-2] + "/" + dir_split_ep_path[-1]
            writer.writerow(
                [
                    short_ep_path.split(".")[0],
                    ep_data[ep_path]["task_prompt"],
                    ep_data[ep_path]["scene_id"],
                ]
            )
    print(f"Wrote successful episode paths to {csv_out_file}")


####################################################
# for quick basic stats run:
# python examples/hitl/isaac_robot_teleop/record_post_process.py --session-records-dir <my_dir> --start-version 0.1.0 --quick
# for stats on local headless execution run:
# python examples/hitl/isaac_robot_teleop/record_post_process.py --session-records-dir "hitl_output_18000" --local
####################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Read and process a directory containing teleop session records and compute statistics."
    )
    parser.add_argument(
        "--session-records-dir",
        type=str,
    )
    parser.add_argument(
        "--out-dir", type=str, default="hitl_teleop_record_stats_out/"
    )
    parser.add_argument("--start-version", type=str, default=None)
    parser.add_argument("--end-version", type=str, default=None)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="If passed, don't process the frame json data, instead only look at the metadata headers. Much faster, but no per-frame data like events.",
    )
    args = parser.parse_args()

    process_record_stats(
        data_dir=args.session_records_dir,
        out_dir=args.out_dir,
        start_version=args.start_version,
        end_version=args.end_version,
        is_local_dir=args.local,
        do_not_process_frames=args.quick,
    )

    # NOTE: this function is meant to print the robot's joint limits re-organized into hardware vector format.
    # get_robot_joint_limits_for_hardware()
