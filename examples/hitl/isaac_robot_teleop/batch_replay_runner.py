import json
import os
import subprocess
import threading
import time

from tqdm import tqdm


def capture_window(window_id, x, y, width, height, output_file, stop_event):
    cmd = [
        "ffmpeg",
        "-f",
        "x11grab",
        "-framerate",
        "30",
        "-video_size",
        f"{width}x{height}",
        "-i",
        f":0.0+{x},{y}",
        output_file,
    ]
    process = subprocess.Popen(cmd)
    stop_event.wait()
    process.terminate()
    process.wait()


def get_window_geometry(window_id):
    cmd = ["xwininfo", "-id", window_id]
    output = subprocess.check_output(cmd).decode()
    geometry = {}
    for line in output.splitlines():
        if "Absolute upper-left X:" in line:
            geometry["x"] = int(line.split(":")[1].strip())
        elif "Absolute upper-left Y:" in line:
            geometry["y"] = int(line.split(":")[1].strip())
        elif "Width:" in line:
            geometry["width"] = int(line.split(":")[1].strip())
        elif "Height:" in line:
            geometry["height"] = int(line.split(":")[1].strip())
    return geometry


def get_window_id(process):
    # Wait for the window to appear
    time.sleep(5)
    cmd = ["xdotool", "search", "--onlyvisible", "--pid", str(process.pid)]
    window_id = subprocess.check_output(cmd).decode().strip()
    return window_id


# NOTE: added this flag because we now have embedded video capture
record_screen = False

# define the range of episodes to consider and details from that batch
episode_dataset = "data/datasets/hitl_teleop_episodes.json.gz"
# session_dir = "../../../Downloads/download/isaac_robot_teleop/"
session_dir = "../../../Downloads/isaac_robot_teleop_vla_b/"

# NOTE: manually set the episode filepaths here
episode_filepaths = [
    "1752951016_1109/12.json.gz",
    "1753051963_9840/1.json.gz",
    "1753051963_9840/0.json.gz",
]

# NOTE: or load a teleop record file
teleop_record_filepath = "hitl_teleop_record_stats_out/hitl_teleop_record.json"
with open(teleop_record_filepath, "rt") as f:
    teleop_record_json = json.load(f)
    episode_filepaths = teleop_record_json["reported_successful_episode_paths"]


script_template = "examples/hitl/isaac_robot_teleop/isaac_robot_teleop.py"

for ep_fp in tqdm(episode_filepaths):
    full_ep_fp = os.path.join(session_dir, ep_fp)
    video_prefix = ep_fp.replace("/", "_").split(".")[0]

    cmd = [
        "python",
        script_template,
        "--config-name",
        "replay_episode_record",
        f"isaac_robot_teleop.episode_dataset={episode_dataset}",
        f"isaac_robot_teleop.episode_record_filepath={full_ep_fp}",
        f"isaac_robot_teleop.replay_speed={20}",
        f"habitat_hitl.target_sps={600}",
        f"isaac_robot_teleop.video_prefix={video_prefix}",
        # NOTE: the following are set in the yaml but added here for ease of edit
        "isaac_robot_teleop.record_video=True",
        # f"isaac_robot_teleop.cam_zoom_distance=1.0",
        # f"isaac_robot_teleop.lock_orientation_to_robot=True",
    ]

    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)

    if record_screen:
        window_id = get_window_id(process)
        geometry = get_window_geometry(window_id)
        output_file = (
            f"hitl_teleop_record_stats_out/video/capture_{video_prefix}.mp4"
        )

        stop_event = threading.Event()
        capture_thread = threading.Thread(
            target=capture_window,
            args=(
                window_id,
                geometry["x"],
                geometry["y"],
                geometry["width"],
                geometry["height"],
                output_file,
                stop_event,
            ),
        )
        capture_thread.start()

    # Wait for the process to finish
    process.wait()
    if record_screen:
        stop_event.set()
        capture_thread.join()
