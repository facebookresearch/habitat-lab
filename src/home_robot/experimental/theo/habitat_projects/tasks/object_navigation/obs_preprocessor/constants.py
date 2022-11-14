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

goal_id_to_goal_name = {
    0: "chair",
    1: "bed",
    2: "potted plant",
    3: "toilet",
    4: "tv",
    5: "couch",
}

goal_id_to_coco_id = {
    0: 0,  # chair
    1: 3,  # bed
    2: 2,  # potted plant
    3: 4,  # toilet
    4: 5,  # tv
    5: 1,  # couch
}

frame_color_palette = [
    *coco_categories_color_palette,
    1.0,
    1.0,
    1.0,  # no category
]
