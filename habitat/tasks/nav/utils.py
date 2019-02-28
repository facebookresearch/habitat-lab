import numpy as np
import cv2
from typing import List, Tuple, Optional


def draw_target_birdseye_view(
    agent_position: np.ndarray,
    agent_rotation: float,
    goal_position: np.ndarray,
    resolution_px: int = 800,
    goal_radius: float = 0.2,
    agent_radius_px: int = 20,
    scales: Optional[List[float]] = None,
    colors: Optional[List[Tuple[int]]] = None,
) -> np.ndarray:
    """Return image of agent w.r.t. centered target location for pointnav tasks.
    """
    if scales is None:
        scales = [15, 10, 5, 2.5, 1]
    if colors is None:
        colors = [
            (47, 19, 122),
            (22, 99, 170),
            (92, 177, 0),
            (226, 169, 0),
            (226, 12, 29),
        ]

    assert len(scales) == len(
        colors
    ), "There must be an equal number of scales and colors."

    goal_agent_dist = np.sqrt(
        np.sum(np.square(agent_position - goal_position))
    )
    goal_distance_padding = max(
        2, 2 ** np.ceil(np.log(goal_agent_dist) / np.log(2))
    )
    movement_scale = 1.0 / goal_distance_padding
    half_res = resolution_px // 2
    im_position = np.full(
        (resolution_px, resolution_px, 3), 255, dtype=np.uint8
    )
    # Draw rings:
    for scale, color in zip(scales, colors):
        if goal_distance_padding * 4 > scale:
            cv2.circle(
                im_position,
                (half_res, half_res),
                max(2, int(half_res * scale * movement_scale)),
                color,
                thickness=-1,
            )

    # Draw such that the agent being inside the radius is the circles overlapping.
    cv2.circle(
        im_position,
        (half_res, half_res),
        max(2, int(half_res * goal_radius * movement_scale) - agent_radius_px),
        (127, 0, 0),
        thickness=-1,
    )

    relative_position = agent_position - goal_position
    relative_position *= half_res * movement_scale
    relative_position += half_res
    relative_position = np.round(relative_position).astype(np.int32)
    rotation_matrix = np.array(
        [
            [np.cos(agent_rotation), -np.sin(agent_rotation)],
            [np.sin(agent_rotation), np.cos(agent_rotation)],
        ]
    )
    arrow_end_point = np.round(
        relative_position[[0, 2]]
        + np.dot(rotation_matrix, np.array([0, half_res * goal_radius / 2]))
    ).astype(np.int32)
    cv2.circle(
        im_position,
        (relative_position[0], relative_position[2]),
        agent_radius_px,
        (0, 0, 255),
        thickness=-1,
    )
    cv2.arrowedLine(
        im_position,
        (relative_position[0], relative_position[2]),
        (arrow_end_point[0], arrow_end_point[1]),
        (0, 255, 0),
        thickness=agent_radius_px // 2,
    )
    # Rotate twice to fix coordinate system to upwards being positive.
    im_position = np.rot90(im_position, 2)
    return im_position
