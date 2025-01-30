import math
import random
from typing import List, Optional, Union

import magnum as mn
import numpy as np
import quaternion

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quat_to_rad, quaternion_rotate_vector


class ShortestPathFollowerv2:
    def __init__(
        self,
        sim: HabitatSim,
        goal_radius: float = 0.7,
    ):
        self._sim = sim
        self.goal_radius = goal_radius

    def get_yaw_from_matrix(self, rotation_matrix):
        return np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    def wrap_heading(self, heading):
        return (heading + np.pi) % (2 * np.pi) - np.pi

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

        return phi

    def shortest_angular_distance(
        self, from_angle: float, to_angle: float
    ) -> float:
        """
        Compute the shortest angular distance between two angles.
        Both angles should be in radians.
        Returns the signed shortest angular distance.
        """
        # Normalize both angles to [-pi, pi]
        from_angle = self.wrap_heading(from_angle)
        to_angle = self.wrap_heading(to_angle)

        # Compute the difference
        error = to_angle - from_angle

        # Normalize the error to [-pi, pi]
        error = self.wrap_heading(error)

        return error

    def heading_error(
        self,
        agent_position: np.ndarray,
        position: np.ndarray,
        agent_heading: np.ndarray,
    ) -> float:
        heading_to_waypoint = np.arctan2(
            position[2] - agent_position[2], position[0] - agent_position[0]
        )
        print("heading_to_waypoint: ", np.rad2deg(heading_to_waypoint))
        agent_heading = self.wrap_heading(agent_heading)
        heading_error = self.wrap_heading(heading_to_waypoint - agent_heading)
        return heading_error

    def get_next_action(self, goal_pos):
        # Convert inputs to numpy arrays if they aren't already
        robot_pose = self._sim.articulated_agent.base_transformation
        robot_pos_YZX = robot_pose.translation
        robot_pos_XY = np.array([-robot_pos_YZX[2], robot_pos_YZX[0]])
        goal_pos_XY = np.array([-goal_pos[2], goal_pos[0]])
        robot_rot_mn_quat = mn.Quaternion.from_matrix(robot_pose.rotation())
        robot_rot_quat = quaternion.quaternion(
            robot_rot_mn_quat.scalar, *robot_rot_mn_quat.vector
        )

        robot_orientation = self.get_yaw_from_matrix(robot_pose.rotation())

        heading_err = self.heading_error(
            robot_pos_YZX, goal_pos, robot_orientation
        )
        print(
            "robot_orientation: ",
            np.rad2deg(robot_orientation),
            "heading_err: ",
            np.rad2deg(heading_err),
        )

        if heading_err > np.deg2rad(30):
            return [0.0, -1.0]
        elif heading_err < -np.deg2rad(30):
            return [0.0, 1.0]
        else:
            distance_to_goal = np.linalg.norm(goal_pos_XY - robot_pos_XY)
            print("distance_to_goal: ", distance_to_goal)
            if distance_to_goal < self.goal_radius:
                return [0.0, 0.0]
            else:
                return [1.0, 0.0]

        # print("robot_orientation_mn: ", robot_orientation_mn)
        # robot_orientation = self._quat_to_xy_heading(robot_rot_quat)
        # robot_orientation = -quat_to_rad(robot_rot_quat) + np.pi / 2

        # Calculate vector to goal
        goal_vector = goal_pos - robot_pos
        distance_to_goal = np.linalg.norm(goal_vector)

        # Calculate desired heading (angle to goal)

        desired_heading = np.arctan2(goal_vector[1], goal_vector[0])

        heading_error = self.shortest_angular_distance(
            robot_orientation, desired_heading
        )

        # # Calculate heading error (difference between current and desired heading)
        # heading_error = self.wrap_heading(
        #     np.arctan2(
        #         np.sin(desired_heading - robot_orientation),
        #         np.cos(desired_heading - robot_orientation),
        #     )
        # )
        print(
            f"robot_orientation: {np.rad2deg(robot_orientation)}, desired_heading: {np.rad2deg(desired_heading)}, heading_error: {np.rad2deg(heading_error)}"
        )

        # Define threshold for when we consider the robot "aligned" with the goal
        HEADING_THRESHOLD = 0.7  # radians (about 5.7 degrees)

        print("heading error: ", heading_error)
        # If not aligned with goal, only turn (no forward motion)
        if abs(heading_error) > HEADING_THRESHOLD:
            angular_velocity = np.clip(2.0 * heading_error, -1.0, 1.0)
            return [0.0, angular_velocity]

        # If aligned with goal, only move forward (no turning)
        else:
            # If we're within goal radius, stop
            if distance_to_goal < self.goal_radius:
                return [0.0, 0.0]
            else:
                # Linear velocity proportional to distance, but capped at 1.0
                linear_velocity = np.clip(distance_to_goal / 5.0, 0.0, 1.0)
                return [linear_velocity, 0.0]

        # # Calculate heading error (difference between current and desired heading)
        # heading_error = np.arctan2(
        #     np.sin(desired_heading - robot_orientation),
        #     np.cos(desired_heading - robot_orientation),
        # )

        # # Calculate linear velocity
        # print("distance_to_goal: ", distance_to_goal)
        # print("heading_error: ", heading_error)
        # # Slow down as we get closer to goal or when our heading error is large
        # linear_velocity = np.clip(distance_to_goal / 5.0, 0.0, 1.0) * np.cos( heading_error )
        # print("linear_velocity: ", linear_velocity)

        # # Calculate angular velocity
        # # Proportional control on heading error
        # angular_velocity = np.clip(2.0 * heading_error, -1.0, 1.0)

        # return [np.abs(linear_velocity), angular_velocity]


# class ShortestPathFollowerv2:
#     def __init__(
#         self, sim: HabitatSim, target_position: Union[List[float], np.ndarray]
#     ):
#         self._sim = sim
#         self.target_position = target_position

#     def step(self):
#         """
#         Step the shortest path follower. Objects will be moved in a point-turn
#         manner (i.e. they won't move forward until they have turned to face
#         the next waypoint).
#         :return:
#         """
#         waypoint = self.target_position
#         translation = self._sim.get_translation(self.object_id)
#         magnum_quaternion = self._sim.get_rotation(self.object_id)

#         # Face the next waypoint if we aren't already facing it
#         if not self.done_turning:
#             # Get current global heading
#             numpy_quaternion = np.quaternion(
#                 magnum_quaternion.scalar, *magnum_quaternion.vector
#             )
#             heading = -quat_to_rad(numpy_quaternion) + np.pi / 2

#             # Get heading necessary to face next waypoint
#             # In habitat, syntax is (x, z, y), for some reason...
#             theta = math.atan2(
#                 waypoint[2] - translation[2], waypoint[0] - translation[0]
#             )
#             theta_diff = get_heading_error(heading, theta)
#             direction = 1 if theta_diff < 0 else -1

#             # If turning at max speed for the entire time step would overshoot,
#             # only turn at the speed necessary to face the waypoint by the end
#             # of the time step. Added a buffer of 20% percent to avoid
#             # very small pivots that waste time.
#             if self.ang_speed * self.time_step * 1.2 >= abs(theta_diff):
#                 angular_velocity = -theta_diff / self.time_step
#                 self.done_turning = True
#             else:
#                 angular_velocity = self.ang_speed * direction

#             linear_velocity = 0.0

#         # If we ARE facing the next waypoint, then move forward
#         else:
#             # If next move would normally overshoot, move just the right amount
#             distance = np.sqrt(
#                 (translation[0] - waypoint[0]) ** 2
#                 + (translation[2] - waypoint[2]) ** 2
#             )

#             # If moving forward at max speed for the entire time step would
#             # overshoot, only move at the speed necessary to reach the waypoint
#             # by the end of the time step. Added a buffer of 20% percent to
#             # avoid very small moves that waste time.
#             if self.max_linear_vel * self.time_step * 1.2 >= distance:
#                 linear_velocity = distance / self.time_step
#                 self.done_turning = False  # start turning to next waypoint
#                 self.next_waypoint_idx += 1
#             else:
#                 linear_velocity = self.max_linear_vel

#             angular_velocity = 0.0
#         return np.array([linear_velocity, angular_velocity])
