import cv2
import numpy as np
import open3d as o3d
import trimesh.transformations as tra


class Camera(object):
    def __init__(
        self,
        pos,
        orn,
        height,
        width,
        fx,
        fy,
        px,
        py,
        near_val,
        far_val,
        pose_matrix,
        proj_matrix,
        view_matrix,
        fov,
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
        info["proj_matrix"] = self.proj_matrix
        info["view_matrix"] = self.view_matrix
        info["max_depth"] = self.max_depth
        info["pose_matrix"] = self.pose_matrix
        info["px"] = self.px
        info["py"] = self.py
        info["fx"] = self.fx
        info["fy"] = self.fy
        info["fov"] = self.fov
        return info

    def get_pose(self):
        return self.pose_matrix.copy()

    def depth_to_xyz(self, depth):
        """get depth from numpy using simple pinhole self model"""
        indices = np.indices((self.height, self.width), dtype=np.float32).transpose(
            1, 2, 0
        )
        z = depth
        # pixel indices start at top-left corner. for these equations, it starts at bottom-left
        x = (indices[:, :, 1] - self.px) * (z / self.fx)
        y = (indices[:, :, 0] - self.py) * (z / self.fy)
        # Should now be height x width x 3, after this:
        xyz = np.stack([x, y, z], axis=-1)
        return xyz

    def fix_depth(self, depth):
        depth = depth.copy()
        depth[depth > self.far_val] = 0
        depth[depth < self.near_val] = 0
        return depth


def z_from_opengl_depth(depth, camera: Camera):
    near = camera.near_val
    far = camera.far_val
    # return (2.0 * near * far) / (near + far - depth * (far - near))
    return (near * far) / (far - depth * (far - near))


def z_from_2(depth, camera: Camera):
    # TODO - remove this?
    height, width = distance.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    x_over_z = (px - camera.px) / camera.fx
    y_over_z = (py - camera.py) / camera.fy
    z = distance / np.sqrt(1.0 + x_over_z**2 + y_over_z**2)
    return z


# We apply this correction to xyz when computing it in sim
# R_CORRECTION = R1 @ R2
T_CORRECTION = tra.euler_matrix(0, 0, np.pi / 2)
R_CORRECTION = T_CORRECTION[:3, :3]


def opengl_depth_to_xyz(depth, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(
        1, 2, 0
    )
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    # indices[..., 0] = np.flipud(indices[..., 0])
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)  # * -1
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1) @ R_CORRECTION
    return xyz


def depth_to_xyz(depth, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(
        1, 2, 0
    )
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def to_o3d_point_cloud(xyz, rgb=None, mask=None):
    """conversion to point cloud from image"""
    xyz = xyz.reshape(-1, 3)
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
    if mask is not None:
        mask = mask.reshape(-1)
        xyz = xyz[mask.astype(np.bool)]
        rgb = rgb[mask.astype(np.bool)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def show_point_cloud(xyz, rgb=None, orig=None, R=None):
    # http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html
    pcd = to_o3d_point_cloud(xyz, rgb)
    geoms = [pcd]
    if orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=orig)
        if R is not None:
            coords = coords.rotate(R)
        geoms.append(coords)
    o3d.visualization.draw_geometries(geoms)


def smooth_mask(mask, kernel=None):
    if kernel is None:
        kernel = np.ones((5, 5))
    mask = mask.astype(np.uint8)
    # h, w = mask.shape[:2]
    mask = cv2.dilate(mask, kernel, iterations=3)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # second step
    mask2 = mask
    mask2 = cv2.erode(mask2, kernel, iterations=3)
    # mask2 = cv2.dilate(mask2, kernel, iterations=1)
    mask2 = np.bitwise_and(mask, mask2)
    return mask, mask2
