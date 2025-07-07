#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import gzip
import json
import math
import os
import pprint
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import magnum as mn
from tqdm import tqdm

# NOTE: use known version days to split sessions by expected features or contents
version_days = {
    "v0.1": (datetime.date(2025, 6, 10), "task prompts and bug fixes"),
    "v0.2": (
        datetime.date(2025, 6, 17),
        "collision detection and IK clamping",
    ),
}

# format of the strings changed mid-pilot, so we'll need to merge data
rollout_dir_format = "timestamp_first"
# rollout_dir_format = "timestamp_last"


def load_json_gz(file_path: str) -> Dict[Any, Any]:
    """
    Return a Dict with the contents of the json.gz file.
    """
    with gzip.open(file_path, "rt") as f:
        return json.load(f)


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


def get_good_first_ep_frame(ep_frames_json: List[Dict[Any, Any]]) -> int:
    """
    From an episode frames JSON dict finds a good first frame for replaying relevant content.
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

    prev_frame_data: Dict[Any, Any] = None
    for ix, frame_data in enumerate(ep_frames_json):
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
        if (
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
) -> None:
    """
    Process a directory containing teleop session records and compute statistics.

    :param data_dir: The directory containing all the session records.
    :param out_dir: The desired relative directory path where output should be saved.
    :param start_version: (optional) The first app version to include in stats calculations.
    :param end_version: (optional) The last app version to include in stats calculations.
    """

    # create the output directory
    os.makedirs(out_dir, exist_ok=True)

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
    for s_dir in session_dirs:
        session_day = convert_timestamp(int(s_dir.split("_")[-1])).date()
        user = s_dir.split("_")[-2]
        users[user] += 1
        session_days[session_day].append(s_dir)

    print("\nANALYSIS:\n")

    print("\nSessions by date:")
    pprint.pprint(session_days, width=25)  # Adjust the width as needed

    print("\nSessions per user:")
    pprint.pprint(users)

    # get all the episodes from the sessions
    session_episodes = [
        os.path.join(data_dir, s_dir, ep)
        for s_dir in session_dirs
        for ep in get_episodes_from_session(os.path.join(data_dir, s_dir))
    ]
    pprint.pprint(session_episodes)
    # count the number of trials for each episode and sort on failures/successes
    fail_eps = []
    success_eps = []
    ep_counts: defaultdict = defaultdict(int)
    frame_count_sum = 0

    # NOTE: do all the stats processing for the episodes here
    print(f"Processing {len(session_episodes)} episode records ...")
    for ep_path in tqdm(session_episodes):
        ep_index = int(ep_path.split("/")[-1].split(".json.gz")[0])
        ep_counts[ep_index] += 1

        # get the record dict from JSON
        ep_json = load_json_gz(ep_path)

        interesting_frames = int(
            ep_json["episode"]["frame_count"]
        ) - get_good_first_ep_frame(ep_json["frames"])
        frame_count_sum += interesting_frames

        # success sorting
        success = get_ep_success(ep_json)
        if success:
            success_eps.append(ep_path)
        else:
            fail_eps.append(ep_path)

    avg_frames = frame_count_sum / len(session_episodes)
    # print("Episode frequency:")
    # pprint.pprint(ep_counts)
    print("Successful Episodes:")
    pprint.pprint(success_eps)

    print(f"Successful: {len(success_eps)}")
    print(f"Failed: {len(fail_eps)}")
    print(f"Success rate: {len(success_eps)/len(session_episodes)}")
    print(f"Frames: total={frame_count_sum}, avg={avg_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Read and process a directory containing teleop session records and compute statistics."
    )
    parser.add_argument(
        "--session-records-dir",
        type=str,
    )
    parser.add_argument("--out-dir", type=str, default="record_stats_out/")
    parser.add_argument("--start-version", type=str, default=None)
    parser.add_argument("--end-version", type=str, default=None)
    args = parser.parse_args()

    process_record_stats(
        data_dir=args.session_records_dir,
        out_dir=args.out_dir,
        start_version=args.start_version,
        end_version=args.end_version,
    )
