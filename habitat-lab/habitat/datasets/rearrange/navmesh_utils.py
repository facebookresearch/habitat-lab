import magnum as mn
import numpy as np

import habitat_sim
from habitat.tasks.utils import get_angle
from habitat_sim.physics import VelocityControl


def is_collision(sim, trans, navmesh_offset) -> bool:
    """
    The function checks if the agent collides with the object
    given the navmesh
    """
    nav_pos_3d = [np.array([xz[0], 0.0, xz[1]]) for xz in navmesh_offset]
    cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
    cur_pos = [
        np.array([xz[0], trans.translation[1], xz[2]]) for xz in cur_pos
    ]

    for pos in cur_pos:  # noqa: SIM110
        # Return true if the pathfinder says it is not navigable
        if not sim.pathfinder.is_navigable(pos):
            return True

    return False


def compute_turn(rel, turn_vel, robot_forward):
    """
    Computing the turnning velocity given the relative position
    """
    is_left = np.cross(robot_forward, rel) > 0
    if is_left:
        vel = [0, -turn_vel]
    else:
        vel = [0, turn_vel]
    return vel


class SimpleVelocityControlEnv:
    """
    A simple environment to control the velocity of the robot
    """

    def __init__(self, sim_freq=120.0):
        # the velocity control
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._sim_freq = sim_freq

    def act(self, trans, vel):
        linear_velocity = vel[0]
        angular_velocity = vel[1]
        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3(
            [linear_velocity, 0.0, 0.0]
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0]
        )
        # Compute the rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        # Get the target rigit state based on the simulation frequency
        target_rigid_state = self.vel_control.integrate_transform(
            1 / self._sim_freq, rigid_state
        )
        # Get the ending pos of the agent
        end_pos = target_rigid_state.translation
        # Offset the height
        end_pos[1] = trans.translation[1]
        # Construct the target trans
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            target_rigid_state.translation,
        )

        return target_trans


def is_navigable_given_robot_navmesh(
    sim,
    start_pos,
    goal_pos,
    navmesh_offset=([0, 0], [0.25, 0], [-0.25, 0]),
    angle_threshold=0.1,
    angular_velocity=1.0,
    distance_threshold=0.5,
    linear_velocity=10.0,
):
    """
    Return True if the robot can navigate from point A
    to point B given the configuration of the navmesh.
    """

    # Get the path finder
    pf = sim.pathfinder
    # Init the shortest path
    path = habitat_sim.ShortestPath()
    # Set the locations
    path.requested_start = start_pos
    path.requested_end = goal_pos
    # Find the path
    pf.find_path(path)
    curr_path_points = path.points
    if len(curr_path_points) == 0:
        return 1.0
    # Set the initial position
    trans = mn.Matrix4(sim.agents[0].scene_node.transformation)
    trans.translation = curr_path_points[0]
    # Get the robot position
    robot_pos = np.array(trans.translation)
    # Get the navigation target
    final_nav_targ = np.array(curr_path_points[-1])
    obj_targ_pos = np.array(curr_path_points[-1])
    # the velocity control
    vc = SimpleVelocityControlEnv()

    at_goal = False

    collision = []

    while not at_goal:
        # Find the path
        path.requested_start = robot_pos
        path.requested_end = goal_pos
        pf.find_path(path)
        curr_path_points = path.points
        cur_nav_targ = curr_path_points[1]
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(trans.transform_vector(forward))
        # Compute relative target
        rel_targ = cur_nav_targ - robot_pos

        # Compute heading angle (2D calculation)
        robot_forward = robot_forward[[0, 2]]
        rel_targ = rel_targ[[0, 2]]
        rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
        # Get the angles
        angle_to_target = get_angle(robot_forward, rel_targ)
        angle_to_obj = get_angle(robot_forward, rel_pos)
        # Compute the distance
        dist_to_final_nav_targ = np.linalg.norm(
            (final_nav_targ - robot_pos)[[0, 2]]
        )
        at_goal = (
            dist_to_final_nav_targ < distance_threshold
            and angle_to_obj < angle_threshold
        )

        if not at_goal:
            if dist_to_final_nav_targ < distance_threshold:
                # Do not want to look at the object to reduce collision
                vel = [0, 0]
                at_goal = True
            elif angle_to_target < angle_threshold:
                # Move towards the target
                vel = [linear_velocity, 0]
            else:
                # Look at the target waypoint.
                vel = compute_turn(rel_targ, angular_velocity, robot_forward)
        else:
            vel = [0, 0]

        trans = vc.act(trans, vel)
        robot_pos = trans.translation
        collision.append(is_collision(sim, trans, navmesh_offset))

    return np.average(collision)


def is_accessible(sim, point, nav_to_min_distance) -> bool:
    """
    Return True if the point is within a threshold distance of the nearest
    navigable point and that the nearest navigable point is on the same
    navigation mesh.

    Note that this might not catch all edge cases since the distance is
    based on Euclidean distance. The nearest navigable point may be
    separated from the object by an obstacle.
    """
    if nav_to_min_distance == -1:
        return True
    snapped = sim.pathfinder.snap_point(point)
    # island_idx: int = sim.pathfinder.get_island(snapped)
    dist = float(np.linalg.norm(np.array((snapped - point))[[0, 2]]))
    return (
        dist
        < nav_to_min_distance
        # and island_idx == sim.navmesh_classification_results["active_island"]
    )
