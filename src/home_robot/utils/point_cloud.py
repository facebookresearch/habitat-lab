import numpy as np
import io
import imageio
import h5py
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh.transformations as tra
import cv2


def get_pcd(xyz, rgb=None):
    xyz = xyz.reshape(-1, 3)
    if rgb is not None:
        rgb = rgb.reshape(-1, 3) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def show_point_cloud(xyz, rgb=None, orig=None, R=None, save=None, grasps=[]):
    # http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
    #    if np.any(rgb > 1):
    #        print("WARNING: rgb values too high! Normalizing...")
    #        rgb = rgb / np.max(rgb)

    pcd = get_pcd(xyz, rgb)
    show_pcd(pcd, orig, R, save, grasps)


def show_pcd(pcd, orig=None, R=None, save=None, grasps=[]):
    geoms = [pcd]
    if orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=orig
        )
        if R is not None:
            coords = coords.rotate(R)
        geoms.append(coords)
    for grasp in grasps:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=grasp[:3, 3]
        )
        coords = coords.rotate(grasp[:3, :3])
        geoms.append(coords)
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    for geom in geoms:
        viz.add_geometry(geom)
    viz.run()
    if save:
        viz.capture_screen_image(save, True)
    viz.destroy_window()


def fix_opengl_image(rgb, depth, camera_params=None):
    """opengl images are weird, we need to do a couple things here"""
    rgb = np.flip(rgb, axis=0)
    depth = np.flip(depth, axis=0)
    if camera_params is not None:
        depth = np.clip(depth, camera_params["near_val"], camera_params["far_val"])
    return rgb, depth


def depth_to_z(depth, near, far):
    """convert depth to metric depth - based on near/far values from camera"""
    depth = 2.0 * depth - 1.0
    z = 2.0 * near * far / (far + near - depth * (far - near))
    return z


def sim_depth_to_world_xyz(depth, width, height, view_matrix, proj_matrix):
    """get points from a camera image rendered in opengl. designed to work with pybullet
    from here: https://github.com/bulletphysics/bullet3/issues/1924#issuecomment-1091876325
    """
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth value
    y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    y *= -1.0
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    # pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]
    # show_point_cloud(points)
    # import pdb; pdb.set_trace()
    return points


def z_from_opengl_depth(depth, camera):
    near = camera.near_val
    far = camera.far_val
    # return (2.0 * near * far) / (near + far - depth * (far - near))
    return (near * far) / (far - depth * (far - near))


def z_from_2(depth, camera):
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
R_CORRECTION = tra.euler_matrix(0, 0, np.pi / 2)[:3, :3]


def opengl_depth_to_xyz(depth, camera):
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


def depth_to_xyz(depth, camera):
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


def build_matrix_of_indices(height, width):
    """Builds a [height, width, 2] numpy array containing coordinates.
    @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)


def pose_from_camera_params(camera_params):
    look_at = camera_params["look_at"]
    look_from = camera_params["look_from"]
    up_vector = camera_params["up_vector"]
    pose = np.eye(4)
    pose[:3, 3] = look_from
    pose[:3, 2] = up_vector
    return pose


def pb_compute_xyz(depth_img, camera_params):
    """Compute ordered point cloud from depth image and camera parameters.
    Assumes camera uses left-handed coordinate system, with
        x-axis pointing right
        y-axis pointing up
        z-axis pointing "forward"
    @param depth_img: a [H x W] numpy array of depth values in meters
    @param camera_params: a dictionary with parameters of the camera used
    @return: a [H x W x 3] numpy array

    from chris xie:
    https://github.com/chrisdxie/uois/blob/master/src/util/utilities.py#L204
    https://github.com/chrisdxie/uois/blob/master/LICENSE
    """

    # Compute focal length from camera parameters
    if "fx" in camera_params and "fy" in camera_params:
        fx = camera_params["fx"]
        fy = camera_params["fy"]
    else:  # simulated data
        aspect_ratio = camera_params["width"] / camera_params["height"]
        e = 1 / (np.tan(np.radians(camera_params["fov"] / 2.0)))
        t = camera_params["near_val"] / e
        b = -t
        r = t * aspect_ratio
        l = -r
        alpha = camera_params["width"] / (r - l)  # pixels per meter
        focal_length = (
            camera_params["near_val"] * alpha
        )  # focal length of virtual camera (frustum camera)
        fx = focal_length
        fy = focal_length

    if "x_offset" in camera_params and "y_offset" in camera_params:
        x_offset = camera_params["x_offset"]
        y_offset = camera_params["y_offset"]
    else:  # simulated data
        x_offset = camera_params["width"] / 2
        y_offset = camera_params["height"] / 2

    indices = build_matrix_of_indices(camera_params["height"], camera_params["width"])
    indices[..., 0] = np.flipud(
        indices[..., 0]
    )  # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


##### Depth augmentations #####
# from: https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py#L68


def rotate_point_cloud(pcd):
    """
    Rotate point-cloud wrt z-axis by a random angle, anti-clockwise=+ve
    """
    rotation_matrix = tra.euler_matrix(0, 0, 2 * np.pi * np.random.random() - np.pi)
    pcd = tra.transform_points(pcd, rotation_matrix)
    return pcd


def add_multiplicative_noise(depth_img, gamma_shape=10000, gamma_scale=0.0001):
    """Add noise to depth image.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [H x W] set of depth z values
    """
    # TODO(cpaxton): I am pretty sure we do not need to copy here, but verify this isn't
    # causing problems
    # depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    # This will randomly shift around points locally
    multiplicative_noise = np.random.gamma(
        gamma_shape, gamma_scale, size=depth_img.shape
    )
    # Apply this noise to the depth image
    depth_img = multiplicative_noise * depth_img
    return depth_img


def add_additive_noise_to_xyz(
    xyz_img,
    gp_rescale_factor_range=[12, 20],
    gaussian_scale_range=[0.0, 0.003],
    valid_mask=None,
):
    """Add (approximate) Gaussian Process noise to ordered point cloud
    @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    gp_rescale_factor = np.random.randint(
        gp_rescale_factor_range[0], gp_rescale_factor_range[1]
    )
    gp_scale = np.random.uniform(gaussian_scale_range[0], gaussian_scale_range[1])

    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(
        loc=0.0, scale=gp_scale, size=(small_H, small_W, C)
    )
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    if valid_mask is not None:
        # use this to add to the image
        xyz_img[valid_mask, :] += additive_noise[valid_mask, :]
    else:
        xyz_img += additive_noise

    return xyz_img


def dropout_random_ellipses(
    depth_img, dropout_mean, gamma_shape=10000, gamma_scale=0.0001
):
    """Randomly drop a few ellipses in the image for robustness.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(dropout_mean)

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(
        np.where(depth_img > 0)
    ).T  # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(
        nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
    )
    dropout_centers = nonzero_pixel_indices[
        dropout_centers_indices, :
    ]  # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        # mask is always 2d even if input is not
        mask = np.zeros(depth_img.shape[:2])
        mask = cv2.ellipse(
            mask,
            tuple(center[::-1]),
            (x_radius, y_radius),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
        )
        depth_img[mask == 1] = 0

    return depth_img
