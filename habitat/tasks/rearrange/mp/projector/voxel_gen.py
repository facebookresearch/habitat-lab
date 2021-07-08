from collections import deque
from typing import Deque

import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from habitat.tasks.rearrange.mp.projector import PointCloud


class VoxelMapper:
    def __init__(self, debug_disp):
        self.debug_disp = debug_disp
        self.points: Deque = deque(maxlen=10000)
        self.keep_points = 200
        self.first_draw = True
        self.thresh = 2.0
        self.angle = 0
        if self.debug_disp:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")

    def get_voxels(self):
        return list(self.points)

    def compute_voxels(self, depth, sim):
        w, h = depth.shape[:2]
        world_shift = torch.FloatTensor([0, 0, 0])
        vfov = 90
        projector = PointCloud(
            vfov, 1, h, w, world_shift, 0.5, device=torch.device("cpu")
        )

        depth = depth[:, :, 0].astype(np.float32)
        depth *= 10.0
        depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0)

        T = sim.cam_trans
        rot_trans = mn.Matrix4.rotation(mn.Deg(180), mn.Vector3(1.0, 0, 0))

        T = T @ rot_trans
        T = torch.tensor(T).T.unsqueeze(0)

        pc, mask_outliers = projector.forward(depth_var, T)
        dv = depth_var.view((*pc.shape[:-1], 1))

        pc = pc[~mask_outliers]
        dv = dv[~mask_outliers]

        pc = pc.numpy()
        dv = dv.numpy()

        pc = pc[0 : -1 : self.keep_points, :]
        dv = dv[0 : -1 : self.keep_points, :]
        keep = dv.reshape(-1) < self.thresh
        pc = pc[keep]
        dv = dv[keep]

        self.points.extend(np.concatenate([pc, dv], axis=1))

        if self.debug_disp:
            # if False:
            use_pc = np.array(self.points)
            x = use_pc[:, 0]
            y = use_pc[:, 1]
            z = use_pc[:, 2]
            d = use_pc[:, 3]

            plt.ion()
            plt.show()

            self.ax.clear()
            p = self.ax.scatter(x, y, z, c=d, cmap=cm.coolwarm)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")
            if self.first_draw:
                plt.colorbar(p)
                self.first_draw = False

            plt.draw()
            plt.pause(0.001)
