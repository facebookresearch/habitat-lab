import cv2
import numpy as np
import torch
import habitat.utils.geometry_utils as geo_utils

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def quaternion_to_yaw(quat):
    """
    Extract yaw from a quaternion, assuming (x, y, z, w) order.
    """
    x, y, z, w = geo_utils.quaternion_to_list(quat)
    
    # Using standard formula: 
    #   yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + z^2))
    sin_yaw = 2.0 * (w * y + x * z)
    cos_yaw = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return yaw


def angle_difference(a, b):
    """
    Returns the angle difference (a - b) in the range (-pi, pi].
    """
    diff = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return diff

def get_desired_yaw(goal_pos, current_pos):
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]

    return np.arctan2(dy, dx)


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def average_min_distance(goal_feat: torch.Tensor, current_feat: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Computes the average minimum distance between each feature in current_feat and the closest feature in goal_feat.

    Args:
        goal_feat (torch.Tensor): Tensor of shape (BS, NR_FEATURES, NR_CHANNELS)
        current_feat (torch.Tensor): Tensor of shape (BS, NR_FEATURES, NR_CHANNELS)
        k (int): Number of closest matches to consider (default: 1, meaning min distance)

    Returns:
        torch.Tensor: The average of the minimum distances for each feature.
    """
    # Compute pairwise Euclidean distances between current_feat and goal_feat
    diff = current_feat.unsqueeze(2) - goal_feat.unsqueeze(1)  # Shape: (BS, NR_FEATURES, NR_FEATURES, NR_CHANNELS)
    dist = torch.norm(diff, dim=-1)  # Shape: (BS, NR_FEATURES, NR_FEATURES)
    
    # Find the k closest goal features for each current feature
    min_distances, _ = torch.topk(dist, k, dim=-1, largest=False)
    
    # Compute the mean over the selected min distances
    avg_min_dist = min_distances.mean()
    
    return avg_min_dist