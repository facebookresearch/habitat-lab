import torch
from torch import Tensor


class MapSizeParameters:
    def __init__(self, resolution, map_size_cm, global_downscaling):
        self.resolution = resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution


def init_map_and_pose_for_env(
    e: int,
    local_map: Tensor,
    global_map: Tensor,
    local_pose: Tensor,
    global_pose: Tensor,
    lmb: Tensor,
    origins: Tensor,
    map_size_parameters: MapSizeParameters,
):
    """Initialize global and local map and sensor pose variables
    for a given environment.
    """
    p = map_size_parameters
    global_pose[e].fill_(0.0)
    global_pose[e, :2] = p.global_map_size_cm / 100.0 / 2.0

    # Initialize starting agent locations
    x, y = (global_pose[e, :2] * 100 / p.resolution).int()
    global_map[e].fill_(0.0)
    global_map[e, 2:4, y - 1 : y + 2, x - 1 : x + 2] = 1.0

    recenter_local_map_and_pose_for_env(
        e,
        local_map,
        global_map,
        local_pose,
        global_pose,
        lmb,
        origins,
        map_size_parameters,
    )


def recenter_local_map_and_pose_for_env(
    e: int,
    local_map: Tensor,
    global_map: Tensor,
    local_pose: Tensor,
    global_pose: Tensor,
    lmb: Tensor,
    origins: Tensor,
    map_size_parameters: MapSizeParameters,
):
    """Re-center local map by updating its boundaries, origins, and
    content, and the local pose for a given environment.
    """
    p = map_size_parameters
    global_loc = (global_pose[e, :2] * 100 / p.resolution).int()
    lmb[e] = get_local_map_boundaries(global_loc, map_size_parameters)
    origins[e] = torch.tensor(
        [
            lmb[e][2] * p.resolution / 100.0,
            lmb[e][0] * p.resolution / 100.0,
            0.0,
        ]
    )
    local_map[e] = global_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
    local_pose[e] = global_pose[e] - origins[e]


def get_local_map_boundaries(
    global_loc: torch.IntTensor, map_size_parameters: MapSizeParameters
) -> torch.IntTensor:
    """Get local map boundaries from global sensor location."""
    p = map_size_parameters
    x, y = global_loc
    device, dtype = global_loc.device, global_loc.dtype

    if p.global_downscaling > 1:
        y1, x1 = y - p.local_map_size // 2, x - p.local_map_size // 2
        y2, x2 = y1 + p.local_map_size, x1 + p.local_map_size

        if y1 < 0:
            y1 = torch.tensor(0, device=device, dtype=dtype)
            y2 = torch.tensor(p.local_map_size, device=device, dtype=dtype)
        if y2 > p.global_map_size:
            y1 = torch.tensor(
                p.global_map_size - p.local_map_size, device=device, dtype=dtype
            )
            y2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)

        if x1 < 0:
            x1 = torch.tensor(0, device=device, dtype=dtype)
            x2 = torch.tensor(p.local_map_size, device=device, dtype=dtype)
        if x2 > p.global_map_size:
            x1 = torch.tensor(
                p.global_map_size - p.local_map_size, device=device, dtype=dtype
            )
            x2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)

    else:
        y1 = torch.tensor(0, device=device, dtype=dtype)
        y2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)
        x1 = torch.tensor(0, device=device, dtype=dtype)
        x2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)

    return torch.stack([y1, y2, x1, x2])
