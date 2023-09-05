# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import random

from . import calc, quaternion, constants, conversions, utils


class Joint(object):
    """Defines a joint. A hierarchy of joints form a skeleton.

    Joint object stores information about child/parent joints, base position
    transforms, and additional information in a dictionary. 

    Attributes:
        name: Optional: Name of the joint. By default, we assign a randomized
            name to the joint.
        dof: Optional: Number of degrees of freedom. By default, we assume a
            fixed ball joint (dof=3). This information is stored in an
            unstructured `info` dictionary.
        parent_joint: Joint object that is parent in the skeleton.
            Set None if joint is root.
        xform_from_parent_joint: Transformation matrix indicating position and
            orientation of joint relative to parent in the character's base
            position. Defaults to identity rotation and zero position offset.
        xform_global: Transformation matrix indicating global position and
            orientation of the joint. xform_global is calculated automatically
            when parent joint is set.
        child_joints: List of child joints. Use `add_child_joints` to add joints
            to this list.
    """

    def __init__(
        self,
        name=None,
        dof=3,
        xform_from_parent_joint=constants.eye_T(),
        parent_joint=None,
        limits=None,
        direction=None,
        length=None,
        axis=None,
    ):
        self.name = name if name else f"joint_{random.getrandbits(32)}"
        self.child_joints = []
        self.index_child_joint = {}
        self.xform_global = constants.eye_T()
        self.xform_from_parent_joint = xform_from_parent_joint
        self.parent_joint = self.set_parent_joint(parent_joint)
        self.info = {"dof": dof}  # set ball joint by default

        self.length = length

        if axis is not None:
            axis = np.deg2rad(axis)
            self.C = conversions.E2R(axis)
            self.Cinv = np.linalg.inv(self.C)
            self.matrix = None
            self.degree = np.zeros(3)
            self.coordinate = None
        if direction is not None:
            self.direction = direction.squeeze()
        if limits is not None:
            self.limits = np.zeros([3, 2])
            for lm, nm in zip(limits, dof):
                if nm == "rx":
                    self.limits[0] = lm
                elif nm == "ry":
                    self.limits[1] = lm
                else:
                    self.limits[2] = lm

    def get_child_joint(self, key):
        return self.child_joints[utils.get_index(self.index_child_joint, key)]

    def get_child_joint_recursive(self):
        """
        This could have duplicated joints if there exists loops in the chain
        """
        joints = []
        for j in self.child_joints:
            joints.append(j)
            joints += j.get_all_child_joint()
        return joints

    def add_child_joint(self, joint):
        assert isinstance(joint, Joint)
        assert joint.name not in self.index_child_joint.keys()
        self.index_child_joint[joint.name] = len(self.child_joints)
        self.child_joints.append(joint)
        joint.set_parent_joint(self)

    def set_parent_joint(self, joint):
        if joint is None:
            self.parent_joint = None
            return
        assert isinstance(joint, Joint)
        self.parent_joint = joint
        self.xform_global = np.dot(
            self.parent_joint.xform_global, self.xform_from_parent_joint,
        )

    def __eq__(self, other):
        if (self.xform_from_parent_joint != other.xform_from_parent_joint).any():
            return False
        if (self.xform_global != other.xform_global).any():
            return False
        if "dof" in self.info and "dof" in other.info:
            if self.info["dof"] != other.info["dof"]:
                return False
        if len(self.child_joints) != len(other.child_joints):
            return False
        return True


class Skeleton(object):
    """Defines a skeleton. A hierarchy of joints form a skeleton.

    Attributes:
        name: Optional: Name of the joint. By default, we assign "skeleton" as
            the name.
        v_up: Defines the up vector of the skeleton. Defaults to y-axis.
        v_face: Defines the facing direction of the skeleton in its base pose.
            Defaults to z-axis.
        v_up_env: Defines the up-vector of the environment in which motion data
            was recorded. Defaults to y-axis.

        Use `add_joint` method to add joints to the Skeleton object so that
        relevant attributes are populated appropriately.

        In the default setting, the skeleton and the environment use y-axis as
        the up-vector, and x-axis as the initial skeleton facing direction.
    """

    def __init__(
        self,
        name="skeleton",
        v_up=np.array([0.0, 1.0, 0.0]),
        v_face=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 1.0, 0.0]),
    ):
        self.name = name
        self.joints = []
        self.index_joint = {}
        self.root_joint = None
        self.num_dofs = 0
        self.v_up = v_up
        self.v_face = v_face
        self.v_up_env = v_up_env

    def get_index_joint(self, key):
        return utils.get_index(self.index_joint, key)

    def get_joint(self, key):
        return self.joints[self.get_index_joint(key)]

    def add_joint(self, joint, parent_joint):
        if parent_joint is None:
            assert self.num_joints() == 0
            self.root_joint = joint
        else:
            parent_joint = self.get_joint(parent_joint)
            parent_joint.add_child_joint(joint)
        self.index_joint[joint.name] = len(self.joints)
        self.joints.append(joint)
        self.num_dofs += joint.info["dof"]

    def num_joints(self):
        return len(self.joints)

    def num_end_effectors(self):
        self.end_effectors = []
        for j in self.joints:
            if len(j.child_joints) == 0:
                self.end_effectors.append(j)
        return len(self.end_effectors)

    def __eq__(self, other):
        if self.num_joints() != other.num_joints():
            return False
        if sorted(self.index_joint.keys()) != sorted(self.index_joint.keys()):
            return False
        return True


class Pose(object):
    """Defines a pose. A list of poses forms a motion sequence.

    Each pose contains position and orientation of joints, in the form of
    transformation matrices, for a particular time step. The position and
    orientation information are relative to base pose. Pose object also
    stores skeleton information it is associated with, in order to access
    base pose joint data.

    Attributes:
        skel: Skeleton object of the character associated with the pose
        data: Pose data must be provided with an np.array of shape
            (num_joints, 4, 4). The order of joints must be the same as the
            order of joints in `skel.joints`.

    Use the `to_matrix()` to convert the pose object to numpy matrix form,
    and `from_matrix(data, skel)` to convert numpy matrix to pose object. This
    is useful for serializing/deserializing data for batch processing, or to 
    create batched tensor data for ML model inputs.

    Use `get_transform(key, local)` to get joint transformation matrices, in
    either global or local (with respect to parent) form.
    """

    def __init__(self, skel, data=None):
        assert isinstance(skel, Skeleton)
        if data is None:
            data = [constants.eye_T for _ in range(skel.num_joints())]
        assert skel.num_joints() == len(data), "{} vs. {}".format(
            skel.num_joints(), len(data)
        )
        self.skel = skel
        self.data = data

    def get_transform(self, key, local):
        skel = self.skel
        if local:
            return self.data[skel.get_index_joint(key)]
        else:
            joint = skel.get_joint(key)
            T = np.dot(
                joint.xform_from_parent_joint,
                self.data[skel.get_index_joint(joint)],
            )
            while joint.parent_joint is not None:
                T_j = np.dot(
                    joint.parent_joint.xform_from_parent_joint,
                    self.data[skel.get_index_joint(joint.parent_joint)],
                )
                T = np.dot(T_j, T)
                joint = joint.parent_joint
            return T

    def set_transform(self, key, T, local, do_ortho_norm=True):
        if local:
            T1 = T
        else:
            T0 = self.skel.get_joint(key).xform_global
            T1 = np.dot(calc.invertT(T0), T)
        if do_ortho_norm:
            """
            This insures that the rotation part of
            the given transformation is valid
            """
            Q, p = conversions.T2Qp(T1)
            Q = quaternion.Q_op(Q, op=["normalize"])
            T1 = conversions.Qp2T(Q, p)
        self.data[self.skel.get_index_joint(key)] = T1

    def get_root_transform(self):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        return self.get_transform(root_idx, local=False)

    def set_root_transform(self, T, local):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        self.set_transform(root_idx, T, local)

    def get_facing_transform(self):
        d, p = self.get_facing_direction_position()
        z = d
        y = self.skel.v_up_env
        x = np.cross(y, z)
        return conversions.Rp2T(np.array([x, y, z]).transpose(), p)

    def get_facing_position(self):
        d, p = self.get_facing_direction_position()
        return p

    def get_facing_direction(self):
        d, p = self.get_facing_direction_position()
        return d

    def get_facing_direction_position(self):
        R, p = conversions.T2Rp(self.get_root_transform())
        d = np.dot(R, self.skel.v_face)
        d = d - calc.projectionOnVector(d, self.skel.v_up_env)
        p = p - calc.projectionOnVector(p, self.skel.v_up_env)
        return d / np.linalg.norm(d), p

    def set_skeleton(self, skel):
        assert skel.num_joints() == len(self.data)
        self.skel = skel

    def to_matrix(self, local=True):
        """
        Returns pose data in transformation matrix format, with shape
        (num_joints, 4, 4)
        """
        transforms = []
        for joint in self.skel.joints:
            transforms.append(self.get_transform(joint, local))
        return np.array(transforms)

    @classmethod
    def from_matrix(cls, data, skel, local=True):
        """
        Expects pose data in transformation matrix format, with shape
        (num_joints, 4, 4)
        """
        num_joints, T_0, T_1 = data.shape
        assert (
            num_joints == skel.num_joints()
        ), "Data for all joints not provided"
        assert T_0 == 4 and T_1 == 4, (
            "Data not provided in 4x4 transformation matrix format. Use "
            "fairmotion.utils.constants.eye_T() for template identity "
            "matrix"
        )
        pose = cls(skel)
        for joint_id in range(len(skel.joints)):
            pose.set_transform(joint_id, data[joint_id], local)
        return pose

    @classmethod
    def interpolate(cls, pose1, pose2, alpha):
        skel = pose1.skel
        data = []
        for j in skel.joints:
            R1, p1 = conversions.T2Rp(pose1.get_transform(j, local=True))
            R2, p2 = conversions.T2Rp(pose2.get_transform(j, local=True))
            R, p = (
                calc.slerp(R1, R2, alpha),
                calc.lerp(p1, p2, alpha),
            )
            data.append(conversions.Rp2T(R, p))
        return Pose(pose1.skel, data)


class Motion(object):
    """Defines a motion sequence.

    Motion consists of the following components:
    - skeleton: Skeleton object containing information about joints and
        base pose of the subject.
    - poses: List of Pose objects. Each frame of motion is defined by a pose.
    Attributes:
        skel: Skeleton object of the character associated with the pose
        poses: List of pose objects, one for each frame of motion
        fps: Rendering frequency in Hz
        info: Free form dictionary to include more information about the motion
            sequence.

    Use the `to_matrix()` to convert the motion object to numpy matrix form
    with shape (num_frames, num_joints, 4, 4), and `from_matrix(data, skel)` to
    convert numpy matrix to pose object. This is useful for
    serializing/deserializing data for batch processing, or to create batched
    tensor data for ML model inputs.
    """

    def __init__(
        self, name="motion", skel=None, fps=60,
    ):
        self.name = name
        self.skel = skel
        self.poses = []
        self.fps = fps
        self.fps_inv = 1.0 / fps
        self.info = {}

    def clear(self):
        self.poses = []
        self.info = {}

    def set_fps(self, fps):
        self.fps = fps
        self.fps_inv = 1.0 / fps

    def set_skeleton(self, skel):
        self.skel = skel
        for idx in range(len(self.poses)):
            self.poses[idx].set_skeleton(skel)

    def add_one_frame(self, pose_data):
        """Adds a pose at the end of motion object.

        Args:
            pose_data: List of pose data, where each pose 
        """
        self.poses.append(Pose(self.skel, pose_data))

    def frame_to_time(self, frame):
        frame = np.clip(frame, 0, len(self.poses) - 1)
        return frame * self.fps_inv

    def time_to_frame(self, time):
        '''
        Adding small value is necessary to prevent error 
        arised from floating point precision
        '''
        return int(time * self.fps + 1e-05)

    def get_pose_by_frame(self, frame):
        assert frame < self.num_frames(), f"{frame} vs. {self.num_frames()}"
        return self.poses[frame]

    def get_pose_by_time(self, time):
        """
        If specified time is close to an integral multiple of (1/fps), returns
        the pose at that time. Else, returns an interpolated version
        """
        time = np.clip(time, 0, self.length())
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.poses[frame1]

        t1 = self.frame_to_time(frame1)
        t2 = self.frame_to_time(frame2)
        alpha = np.clip((time - t1) / (t2 - t1), 0.0, 1.0)

        return Pose.interpolate(self.poses[frame1], self.poses[frame2], alpha)

    def num_frames(self):
        return len(self.poses)

    def length(self):
        """
        Returns time length of motion in seconds. The first frame is considered
        to be at time 0, and the last frame at time self.length().
        Example: If fps is 60Hz and there are 120 frames, length() returns
        1.9833. In case there are 121 frames, length() is 2.
        """
        return (len(self.poses) - 1) * self.fps_inv

    def to_matrix(self, local=True):
        """
        Returns pose data in transformation matrix format, with shape
        (seq_len, num_joints, 4, 4)
        """
        data = []
        for pose in self.poses:
            data.append(pose.to_matrix(local))
        return np.array(data)

    def rotations(self, local=True):
        """
        Returns joint rotations in rotation matrix format, with shape
        (seq_len, num_joints, 3, 3)
        """
        return self.to_matrix(local)[..., :3, :3]

    def positions(self, local=True):
        """
        Returns joint positions with shape (seq_len, num_joints, 3)
        """
        return self.to_matrix(local)[..., :3, 3]

    @classmethod
    def from_matrix(cls, data, skel, local=True, fps=None):
        """
        Expects pose data in transformation matrix format, with shape
        (seq_len, num_joints, 4, 4)
        """
        assert data.ndim == 4, (
            "Data must be provided in transformation matrix format, with shape"
            " (seq_len, num_joints, 4, 4)"
        )
        seq_len, num_joints, T_0, T_1 = data.shape
        assert (
            num_joints == skel.num_joints()
        ), "Data for all joints not provided"
        assert T_0 == 4 and T_1 == 4, (
            "Data not provided in 4x4 transformation matrix format. Use "
            "fairmotion.utils.constants.eye_T() for template identity "
            "matrix"
        )
        if fps is None:
            fps = 60
        motion = cls(skel=skel, fps=fps)
        for pose_data in data:
            pose = Pose.from_matrix(pose_data, skel, local)
            motion.poses.append(pose)
        return motion
