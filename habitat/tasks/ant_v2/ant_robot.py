import magnum as mn
import numpy as np

from habitat.tasks.ant_v2.quadruped_wrapper import (
    QuadrupedRobot,
    QuadrupedRobotParams,
    RobotCameraParams,
)


class AntV2Robot(QuadrupedRobot):
    def __init__(self, urdf_path, sim, limit_robo_joints=True, fixed_base=False):
        ant_params = QuadrupedRobotParams(
            hip_joints=[6, 11, 16, 1],
            ankle_joints=[8, 13, 18, 3],

            hip_init_params=[0,0,0,0],
            ankle_init_params=[-1,1,1,-1],

            cameras={
                "robot_arm": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                    attached_link_id=22,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(90)),
                ),
                "robot_head": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0.17, 1.2, 0.0),
                    cam_look_at_pos=mn.Vector3(0.75, 1.0, 0.0),
                    attached_link_id=-1,
                ),
                "robot_third": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
            },

            hip_mtr_pos_gain=0.2,
            hip_mtr_vel_gain=0.2,
            hip_mtr_max_impulse=10.0,
            ankle_mtr_pos_gain=0.2,
            ankle_mtr_vel_gain=0.2,
            ankle_mtr_max_impulse=10.0,

            base_offset=mn.Vector3(0,0,0),
            base_link_names={
                "torso",
            },
        )
        super().__init__(ant_params, urdf_path, sim, limit_robo_joints, fixed_base)


    def reconfigure(self) -> None:
        super().reconfigure()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    def reset(self) -> None:
        super().reset()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0))
        return self.sim_obj.transformation @ add_rot

    @property
    def observational_space(self) -> np.ndarray:
        """
        27 dim obs space
        z (height) of the Torso -> 1
        orientation (quarternion x,y,z,w) of the Torso -> 4
        8 Joint angles -> 8
        3-dim directional velocity and 3-dim angular velocity -> 3+3=6
        8 Joint velocity -> 8
        """
        # May expand to make use of external forces in the future (once this is exposed in habitat_sim & if joint torques are used in the future)
        obs_space = np.zeros(27)
        pos = super().base_pos
        obs_space[0] = pos[1]

        # ant orientation
        orientation = super().base_rot
        obs_space[1] = orientation.vector.x
        obs_space[2] = orientation.vector.y
        obs_space[3] = orientation.vector.z
        obs_space[4] = orientation.scalar

        # ant joint angles (Radians)
        obs_space[5:13] = super().leg_joint_pos

        # ant directional velocity
        obs_space[13:16] = super().base_velocity

        # ant angular velocity
        obs_space[16:19] = super().base_angular_velocity

        # ant joint velocity
        obs_space[19:27] = super().joint_velocities
        #obs_space[27:30] = super().base_pos

        
        return obs_space

    def update(self):
        super().update()
