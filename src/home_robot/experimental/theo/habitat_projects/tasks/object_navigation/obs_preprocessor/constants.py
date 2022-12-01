import pandas as pd
from pathlib import Path

from home_robot.agent.perception.detection.coco_maskrcnn.coco_categories import (
    coco_categories_color_palette,
)


MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001

challenge_goal_name_to_goal_name = {
    "chair": "chair",
    "sofa": "couch",
    "plant": "potted plant",
    "bed": "bed",
    "toilet": "toilet",
    "tv_monitor": "tv",
}

hm3d_goal_id_to_goal_name = {
    0: "chair",
    1: "bed",
    2: "potted plant",
    3: "toilet",
    4: "tv",
    5: "couch",
}

hm3d_goal_id_to_coco_id = {
    0: 0,  # chair
    1: 3,  # bed
    2: 2,  # potted plant
    3: 4,  # toilet
    4: 5,  # tv
    5: 1,  # couch
}

floorplanner_goal_id_to_goal_name = {
    1: "bed",
    2: "chair",
    3: "plant",
    4: "couch",
    5: "toilet",
    6: "tv",
}

floorplanner_goal_id_to_coco_id = {
    1: 3,  # bed
    2: 0,  # chair
    3: 2,  # plant
    4: 1,  # couch
    5: 4,  # toilet
    6: 5,  # tv
}

frame_color_palette = [
    *coco_categories_color_palette,
    1.0,
    1.0,
    1.0,  # no category
]

mp3d_to_coco = {
    3: 0,   # chair
    10: 1,  # couch
    14: 2,  # plant
    11: 3,  # bed
    18: 4,  # toilet
    22: 5,  # tv
    5: 6,   # table
    15: 8,  # sink
}

hm3d_to_mp3d_path = Path(__file__).resolve().parent / "matterport_category_mappings.tsv"
df = pd.read_csv(hm3d_to_mp3d_path, sep='    ', header=0)
hm3d_to_mp3d = {row["category"]: row["mpcat40index"] for _, row in df.iterrows()}
