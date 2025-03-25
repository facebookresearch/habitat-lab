import json
from typing import Any, Dict
import os
import magnum as mn
from hydra import compose
from omegaconf import DictConfig

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim  # unfortunately we can't import this earlier

from habitat_sim.gfx import DebugLineRender

# path to this example app directory
dir_path = os.path.dirname(os.path.realpath(__file__)).split("scripts")[0]


def debug_draw_axis(
    dblr: DebugLineRender, transform: mn.Matrix4 = None, scale: float = 1.0
) -> None:
    if transform is not None:
        dblr.push_transform(transform)
    for unit_axis in range(3):
        vec = mn.Vector3()
        vec[unit_axis] = 1.0
        color = mn.Color3(0.5)
        color[unit_axis] = 1.0
        dblr.draw_transformed_line(mn.Vector3(), vec * scale, color)
    if transform is not None:
        dblr.pop_transform()


class Robot:
    """
    Wrapper class for robots imported as simulated ArticulatedObjects.
    Wraps the ManagedObjectAPI.
    """

    def __init__(self, sim: habitat_sim.Simulator, robot_cfg: DictConfig):
        """
        Initialize the robot in a Simulator from its config object.
        """

        self.sim = sim
        self.robot_cfg = robot_cfg
        # expect a "urdf" config field with the filepath
        self.ao = self.sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            self.robot_cfg.urdf,
            fixed_base=self.robot_cfg.fixed_base
            if hasattr(self.robot_cfg, "fixed_base")
            else False,
            force_reload=True,
        )

        # create joint motors
        self.motor_ids_to_link_ids: Dict[int, int] = {}
        if self.robot_cfg.create_joint_motors:
            self.create_joint_motors()

        # set initial pose
        self.set_cached_pose(
            pose_name=self.robot_cfg.initial_pose, set_positions=True
        )
        #self.init_ik()

    def init_ik(self):
        """
        Initialize pymomentum and load a model.
        """
        try:
            import pymomentum.geometry as pym_geo

            self.momentum_character = pym_geo.Character.load_urdf(
                self.robot_cfg.urdf
            )
            # TODO: the above character is available for ik
        except:
            print("Could not initialize pymomentum IK library.")

    def create_joint_motors(self):
        """
        Creates a full set of joint motors for the robot.
        """
        self.motor_settings = habitat_sim.physics.JointMotorSettings(
            0,  # position_target
            self.robot_cfg.joint_motor_pos_gains,  # position_gain
            0,  # velocity_target
            self.robot_cfg.joint_motor_vel_gains,  # velocity_gain
            self.robot_cfg.joint_motor_max_impulse,  # max_impulse
        )

        self.motor_ids_to_link_ids = self.ao.create_all_motors(
            self.motor_settings
        )

    def clean(self) -> None:
        """
        Cleans up the robot. This object is expected to be deleted immediately after calling this function.
        """
        self.sim.get_articulated_object_manager().remove_object_by_handle(
            self.ao.handle
        )

    def place_robot(self, base_pos: mn.Vector3):
        """
        Place the robot at a given position.
        """
        y_size, center = sutils.get_obj_size_along(
            self.sim, self.ao.object_id, mn.Vector3(0, -1, 0)
        )
        offset = (self.ao.translation - center)[1] + y_size
        self.ao.translation = base_pos + mn.Vector3(0, offset, 0)

    def set_cached_pose(
        self,
        pose_file: str = os.path.join(dir_path, "robot_poses.json"),
        pose_name: str = "default",
        set_positions: bool = False,
    ) -> None:
        """
        Loads a robot pose from a json file which could have multiple poses.
        """
        if not os.path.exists(pose_file):
            print(
                f"Cannot load cached pose. Configured pose file {pose_file} does not exist."
            )
            return

        with open(pose_file, "r") as f:
            poses = json.load(f)
            if self.ao.handle not in poses:
                print(
                    f"Cannot load cached pose. No poses cached for robot {self.ao.handle}."
                )
                return
            if pose_name not in poses[self.ao.handle]:
                print(
                    f"Cannot load cached pose. No pose named {pose_name} cached for robot {self.ao.handle}."
                )
                return
            pose = poses[self.ao.handle][pose_name]
            if len(pose) != len(self.ao.joint_positions):
                print(
                    f"Cannot load cached pose (size {len(pose)}) as it does not match number of dofs ({len(self.ao.joint_positions)})"
                )
                return
            if self.robot_cfg.create_joint_motors:
                self.ao.update_all_motor_targets(pose)
            if set_positions:
                self.ao.joint_positions = pose

    def cache_pose(
        self,
        pose_file: str = os.path.join(dir_path, "robot_poses.json"),
        pose_name: str = "default",
    ) -> None:
        """
        Saves the current robot pose in a json cache file with the given name.
        """
        # create the directory if it doesn't exist
        dir = pose_file[: -len(pose_file.split("/")[-1])]
        os.makedirs(dir, exist_ok=True)
        poses = {}
        if os.path.exists(pose_file):
            with open(pose_file, "r") as f:
                poses = json.load(f)
        if self.ao.handle not in poses:
            poses[self.ao.handle] = {}
        poses[self.ao.handle][pose_name] = self.ao.joint_positions
        with open(pose_file, "w") as f:
            json.dump(
                poses,
                f,
                indent=4,
            )

    def draw_debug(self, dblr: DebugLineRender):
        """
        Draw the bounding box of the robot.
        """
        dblr.push_transform(self.ao.transformation)
        bb = self.ao.aabb
        dblr.draw_box(bb.min, bb.max, mn.Color3(1.0, 1.0, 1.0))
        dblr.pop_transform()
        debug_draw_axis(dblr, transform=self.ao.transformation)

        # draw the navmesh circle
        dblr.draw_circle(
            self.ao.translation,
            radius=self.robot_cfg.navmesh_radius,
            color=mn.Color4(0.8, 0.7, 0.9, 0.8),
            normal=mn.Vector3(0, 1, 0),
        )

    def draw_dof(
        self, dblr: DebugLineRender, link_ix: int, cam_pos: mn.Vector3
    ) -> None:
        """
        Draw a visual indication of the given dof state.
        A circle aligned with the dof axis for revolute joints.
        A line with bars representing the min and max joint limits and a bar between them representing state.
        """
        if self.ao.get_link_num_dofs(link_ix) == 0:
            return

        link_obj_id = self.ao.link_ids_to_object_ids[link_ix]
        obj_bb, transform = sutils.get_bb_for_object_id(self.sim, link_obj_id)
        center = transform.transform_point(obj_bb.center())
        size_to_camera, center = sutils.get_obj_size_along(
            self.sim, link_obj_id, cam_pos - center
        )
        draw_at = center + (cam_pos - center).normalized() * size_to_camera

        link_T = self.ao.get_link_scene_node(link_ix).transformation
        global_link_pos = link_T.translation - link_T.transform_vector(
            self.ao.get_link_joint_to_com(link_ix)
        )

        joint_limits = self.ao.joint_position_limits
        joint_positions = self.ao.joint_positions

        for local_dof in range(self.ao.get_link_num_dofs(link_ix)):
            # this link has dofs

            dof = self.ao.get_link_joint_pos_offset(link_ix) + local_dof
            dof_value = joint_positions[dof]
            min_dof = joint_limits[0][dof]
            max_dof = joint_limits[1][dof]
            interp_dof = (dof_value - min_dof) / (max_dof - min_dof)

            j_type = self.ao.get_link_joint_type(link_ix)
            dof_axes = self.ao.get_link_joint_axes(link_ix)
            debug_draw_axis(
                dblr,
                transform=self.ao.get_link_scene_node(link_ix).transformation,
            )
            if j_type == habitat_sim.physics.JointType.Revolute:
                # points out of the rotation plane
                dof_axis = dof_axes[0]
                dblr.draw_circle(
                    global_link_pos,
                    radius=0.1,
                    color=mn.Color3(0, 0.75, 0),  # green
                    normal=link_T.transform_vector(dof_axis),
                )
            elif j_type == habitat_sim.physics.JointType.Prismatic:
                # points along the translation axis
                dof_axis = dof_axes[1]
                # TODO
            # no other options are supported presently

