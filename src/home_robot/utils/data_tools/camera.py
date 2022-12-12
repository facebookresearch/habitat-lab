# Facebook (c) 2022

"""
Load data and visualize it
"""

import os
import json
import numpy as np
import open3d as o3d
import trimesh
import trimesh.transformations as tra
import torch
import pickle


class Camera(object):
    """Camera object storing information about a single camera.
    Includes projection and pose information."""

    def __init__(
        self,
        pos=None,
        orn=None,
        height=None,
        width=None,
        fx=None,
        fy=None,
        px=None,
        py=None,
        near_val=None,
        far_val=None,
        pose_matrix=None,
        proj_matrix=None,
        view_matrix=None,
        fov=None,
        *args,
        **kwargs
    ):
        self.pos = pos
        self.orn = orn
        self.height = height
        self.width = width
        self.px = px
        self.py = py
        self.fov = fov
        self.near_val = near_val
        self.far_val = far_val
        self.fx = fx
        self.fy = fy
        self.pose_matrix = pose_matrix
        self.pos = pos
        self.orn = orn

    def to_dict(self):
        """create a dictionary so that we can extract the necessary information for
        creating point clouds later on if we so desire"""
        info = {}
        info["pos"] = self.pos
        info["orn"] = self.orn
        info["height"] = self.height
        info["width"] = self.width
        info["near_val"] = self.near_val
        info["far_val"] = self.far_val
        # info['proj_matrix'] = self.proj_matrix
        # info['view_matrix'] = self.view_matrix
        # info['max_depth'] = self.max_depth
        info["pose_matrix"] = self.pose_matrix
        info["px"] = self.px
        info["py"] = self.py
        info["fx"] = self.fx
        info["fy"] = self.fy
        info["fov"] = self.fov
        return info

    def get_pose(self):
        return self.pose_matrix.copy()
