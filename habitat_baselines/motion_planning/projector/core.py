import math

import torch


def _transform3D(xyzhe, device=None):
    """Return (N, 4, 4) transformation matrices from (N,5) x,y,z,heading,elevation"""
    if device is None:
        device = torch.device("cpu")

    theta_x = xyzhe[:, 4]  # elevation
    cx = torch.cos(theta_x)
    sx = torch.sin(theta_x)

    theta_y = xyzhe[:, 3]  # heading
    cy = torch.cos(theta_y)
    sy = torch.sin(theta_y)

    T = torch.zeros(xyzhe.shape[0], 4, 4, device=device)
    T[:, 0, 0] = cy
    T[:, 0, 1] = sx * sy
    T[:, 0, 2] = cx * sy
    T[:, 0, 3] = xyzhe[:, 0]  # x

    T[:, 1, 0] = 0
    T[:, 1, 1] = cx
    T[:, 1, 2] = -sx
    T[:, 1, 3] = xyzhe[:, 1]  # y

    T[:, 2, 0] = -sy
    T[:, 2, 1] = cy * sx
    T[:, 2, 2] = cy * cx
    T[:, 2, 3] = xyzhe[:, 2]  # z

    T[:, 3, 3] = 1
    return T


class ProjectorUtils:
    def __init__(
        self,
        vfov,
        batch_size,
        feature_map_height,
        feature_map_width,
        output_height,
        output_width,
        gridcellsize,
        world_shift_origin,
        z_clip_threshold,
        device,
    ):

        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.output_height = output_height  # dimensions of the topdown map
        self.output_width = output_width
        self.gridcellsize = gridcellsize
        self.z_clip_threshold = z_clip_threshold
        self.world_shift_origin = world_shift_origin
        self.device = device

        self.x_scale, self.y_scale, self.ones = self.compute_scaling_params(
            batch_size, feature_map_height, feature_map_width
        )

    def compute_intrinsic_matrix(self, width, height, vfov):
        hfov = width / height * vfov
        f_x = width / (2.0 * math.tan(hfov / 2.0))
        f_y = height / (2.0 * math.tan(vfov / 2.0))
        cy = height / 2.0
        cx = width / 2.0
        K = torch.Tensor([[f_x, 0, cx], [0, f_y, cy], [0, 0, 1.0]])
        return K

    def compute_scaling_params(self, batch_size, image_height, image_width):
        """Precomputes tensors for calculating depth to point cloud"""
        # (float tensor N,3,3) : Camera intrinsics matrix
        K = self.compute_intrinsic_matrix(image_width, image_height, self.vfov)
        K = K.to(device=self.device).unsqueeze(0)
        K = K.expand(batch_size, 3, 3)

        fx = K[:, 0, 0].unsqueeze(1).unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1).unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1).unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1).unsqueeze(1)

        x_rows = torch.arange(start=0, end=image_width, device=self.device)
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((image_height, 1))
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((batch_size, 1, 1))
        x_rows = x_rows.float()

        y_cols = torch.arange(start=0, end=image_height, device=self.device)
        y_cols = y_cols.unsqueeze(1)
        y_cols = y_cols.repeat((1, image_width))
        y_cols = y_cols.unsqueeze(0)
        y_cols = y_cols.repeat((batch_size, 1, 1))
        y_cols = y_cols.float()

        # 0.5 is so points are projected through the center of pixels
        x_scale = (x_rows + 0.5 - cx) / fx  # ; print(x_scale[0,0,:])
        y_scale = (y_cols + 0.5 - cy) / fy  # ; print(y_scale[0,:,0]); stop
        ones = (
            torch.ones(
                (batch_size, image_height, image_width), device=self.device
            )
            .unsqueeze(3)
            .float()
        )
        return x_scale, y_scale, ones

    def point_cloud(self, depth, depth_scaling=1.0):
        """
        Converts image pixels to 3D pointcloud in camera reference using depth values.

        Args:
            depth (torch.FloatTensor): (batch_size, height, width)

        Returns:
            xyz1 (torch.FloatTensor): (batch_size, height * width, 4)

        Operation:
            z = d / scaling
            x = z * (u-cx) / fx
            y = z * (v-cv) / fy
        """
        shape = depth.shape
        if (
            shape[0] == self.batch_size
            and shape[1] == self.fmh
            and shape[2] == self.fmw
        ):
            x_scale = self.x_scale
            y_scale = self.y_scale
            ones = self.ones
        else:
            x_scale, y_scale, ones = self.compute_scaling_params(
                shape[0], shape[1], shape[2]
            )
        z = depth / float(depth_scaling)
        x = z * x_scale
        y = z * y_scale

        # rotate axis to algin with Habitat-MP3D axis definition (y is up)
        # z = -z
        # y = -y
        # Do it using the T transform matrix

        xyz1 = torch.cat(
            (x.unsqueeze(3), y.unsqueeze(3), z.unsqueeze(3), ones), dim=3
        )
        return xyz1

    def transform_camera_to_world(self, xyz1, T):
        """
        Converts pointcloud from camera to world reference.

        Args:
            xyz1 (torch.FloatTensor): [(x,y,z,1)] array of N points in homogeneous coordinates
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)

        Returns:
            (float tensor BxNx4): array of pointcloud in homogeneous coordinates

        Shape:
            Input:
                xyz1: (batch_size, 4, no_of_points)
                T: (batch_size, 4, 4)
            Output:
                (batch_size, 4, no_of_points)

        Operation: T' * R' * xyz
                   Here, T' and R' are the translation and rotation matrices.
                   And T = [R' T'] is the combined matrix provided in the function as input
                           [0  1 ]
        """
        return torch.bmm(T, xyz1)

    def pixel_to_world_mapping(self, depth_img_array, T):
        """
        Computes mapping from image pixels to 3D world (x,y,z)

        Args:
            depth_img_array (torch.FloatTensor): Depth values tensor
            T (torch.FloatTensor): camera-to-world transformation matrix (inverse of
                                        extrinsic matrix)

        Returns:
            pixel_to_world (torch.FloatTensor) : Mapping of one image pixel (i,j) in 3D world
                                                      (x,y,z)
                    array cell (i,j) : (x,y,z)
                        i,j - image pixel indices
                        x,y,z - world coordinates

        Shape:
            Input:
                depth_img_array: (N, height, width)
                T: (N, 4, 4)
            Output:
                pixel_to_world: (N, height, width, 3)
        """

        # Transformed from image coordinate system to camera coordinate system, i.e origin is
        # Camera location  # GEO:
        # shape: xyz1 (batch_size, height, width, 4)
        xyz1 = self.point_cloud(depth_img_array)

        # shape: (batch_size, height * width, 4)
        xyz1 = torch.reshape(
            xyz1, (xyz1.shape[0], xyz1.shape[1] * xyz1.shape[2], 4)
        )

        # shape: (batch_size, 4, height * width)
        xyz1_t = torch.transpose(xyz1, 1, 2)  # [B,4,HxW]

        # Transformed points from camera coordinate system to world coordinate system  # GEO:
        # shape: xyz1_w(batch_size, 4, height * width)
        xyz1_w = self.transform_camera_to_world(xyz1_t, T)

        # shape: (batch_size, height * width, 3)
        world_xyz = xyz1_w.transpose(1, 2)[:, :, :3]

        # -- shift world origin
        world_xyz -= self.world_shift_origin

        # shape: (batch_size, height, width, 3)
        pixel_to_world = torch.reshape(
            world_xyz,
            (
                (
                    depth_img_array.shape[0],
                    depth_img_array.shape[1],
                    depth_img_array.shape[2],
                    3,
                )
            ),
        )

        return pixel_to_world

    def discretize_point_cloud(self, point_cloud, camera_height):
        """#GEO:
        Maps pixel in world coordinates to an (output_height, output_width) map.
        - Discretizes the (x,y) coordinates of the features to gridcellsize.
        - Remove features that lie outside the (output_height, output_width) size.
        - Computes the min_xy and size_xy, and change (x,y) coordinates setting min_xy as origin.

        Args:
            point_cloud (torch.FloatTensor): (x,y,z) coordinates of features in 3D world
            camera_height (torch.FloatTensor): y coordinate of the camera used for deciding
                                                      after how much height to crop

        Returns:
            pixels_in_map (torch.LongTensor): World (x,y) coordinates of features discretized
                                    in gridcellsize and cropped to (output_height, output_width).

        Shape:
            Input:
                point_cloud: (batch_size, features_height, features_width, 3)
                camera_height: (batch_size)
            Output:
                pixels_in_map: (batch_size, features_height, features_width, 2)
        """

        # -- /!\/!\
        # -- /!\ in Habitat-MP3D y-axis is up. /!\/!\
        # -- /!\/!\
        pixels_in_map = (
            (point_cloud[:, :, :, [0, 2]]) / self.gridcellsize
        ).round()

        # Anything outside map boundary gets mapped to (0,0) with an empty feature
        # mask for outside map indices
        outside_map_indices = (
            (pixels_in_map[:, :, :, 0] >= self.output_width)
            + (pixels_in_map[:, :, :, 1] >= self.output_height)
            + (pixels_in_map[:, :, :, 0] < 0)
            + (pixels_in_map[:, :, :, 1] < 0)
        )

        # shape: camera_y (batch_size, features_height, features_width)
        camera_y = (
            camera_height.unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, pixels_in_map.shape[1], pixels_in_map.shape[2])
        )

        # Anything above camera_y + z_clip_threshold will be ignored
        above_threshold_z_indices = point_cloud[:, :, :, 1] > (
            camera_y + self.z_clip_threshold
        )

        mask_outliers = outside_map_indices + above_threshold_z_indices

        return pixels_in_map.long(), mask_outliers
