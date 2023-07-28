from typing import Any, List

import magnum as mn
import numpy as np

import habitat_sim
from habitat.core.logging import logger
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

    def __init__(self, sim_freq=60.0):
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
            [0.0, 0.0, -linear_velocity]
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0]
        )
        # Compute the rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        # Get the target rigid state based on the simulation frequency
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
    navmesh_offset=([0, 0], [0, 0.15], [0, -0.15]),
    angle_threshold=0.05,
    angular_velocity=1.0,
    distance_threshold=0.25,
    linear_velocity=1.0,
    vdb=None,
    render_debug_video=False,
):
    """
    Return True if the robot can navigate from point A
    to point B given the configuration of the navmesh.
    """
    logger.info(
        "Checking robot navigability between target object start and goal:"
    )

    # get the largest island id for pathfinding
    island_areas = list(
        map(
            sim.pathfinder.island_area,
            range(sim.pathfinder.num_islands),
        )
    )
    largest_island_id = island_areas.index(max(island_areas))
    snapped_start_pos = sim.pathfinder.snap_point(start_pos, largest_island_id)
    snapped_goal_pos = sim.pathfinder.snap_point(goal_pos, largest_island_id)

    logger.info(
        f"     - start_pos to snapped_start_pos distance = {(snapped_start_pos-start_pos).length()}"
    )
    logger.info(
        f"     - goal_pos to snapped_goal_pos distance = {(snapped_goal_pos-goal_pos).length()}"
    )

    if render_debug_video:
        assert vdb is not None, "Need a vdb for visual debugging."
        sim.navmesh_visualization = True

    # Create a new pathfinder with slightly stricter radius to provide nav buffer from collision
    pf = habitat_sim.nav.PathFinder()
    modified_settings = sim.pathfinder.nav_mesh_settings
    modified_settings.agent_radius += 0.05
    assert sim.pathfinder.nav_mesh_settings != modified_settings
    assert sim.recompute_navmesh(
        pf, modified_settings
    ), "failed to recompute navmesh"
    # Init the shortest path
    path = habitat_sim.ShortestPath()
    # Set the locations
    path.requested_start = snapped_start_pos
    path.requested_end = snapped_goal_pos
    # Find the path
    pf.find_path(path)
    curr_path_points = path.points
    if len(curr_path_points) <= 2:
        return 1.0
    # Set the initial position
    trans: mn.Matrix4 = mn.Matrix4()
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
    debug_video_frames: List[Any] = []
    debug_framerate = 30
    time_since_debug_frame = 9999.0

    try:
        while not at_goal:
            # Find the path
            path.requested_start = robot_pos
            path.requested_end = snapped_goal_pos
            pf.find_path(path)
            curr_path_points = path.points
            cur_nav_targ = curr_path_points[1]
            forward = np.array([0, 0, -1.0])
            robot_forward = np.array(trans.transform_vector(forward))
            # Compute relative target
            rel_targ = cur_nav_targ - robot_pos
            rel_targ = rel_targ[[0, 2]]

            if np.linalg.norm(rel_targ) < 0.01 and len(curr_path_points) > 2:
                # skip silly turning very close to nav points
                cur_nav_targ = curr_path_points[2]
                rel_targ = cur_nav_targ - robot_pos
                rel_targ = rel_targ[[0, 2]]

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
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
                    vel = compute_turn(
                        rel_targ, angular_velocity, robot_forward
                    )
            else:
                vel = [0, 0]
            trans = vc.act(trans, vel)
            robot_pos = trans.translation
            collision.append(is_collision(sim, trans, navmesh_offset))
            if (
                render_debug_video
                and time_since_debug_frame > 1.0 / debug_framerate
            ):
                time_since_debug_frame = 0
                # render the shortest path
                path_point_render_lines = []
                for i in range(len(curr_path_points)):
                    if i > 0:
                        path_point_render_lines.append(
                            (
                                [curr_path_points[i - 1], curr_path_points[i]],
                                mn.Color4.cyan(),
                            )
                        )
                # render a local axis for the robot
                debug_lines = [
                    (
                        [
                            robot_pos,
                            robot_pos
                            + trans.transform_vector(mn.Vector3(0.3, 0, 0)),
                        ],
                        mn.Color4.red(),
                    ),
                    (
                        [
                            robot_pos,
                            robot_pos
                            + trans.transform_vector(mn.Vector3(0, 0.3, 0)),
                        ],
                        mn.Color4.green(),
                    ),
                    (
                        [
                            robot_pos,
                            robot_pos
                            + trans.transform_vector(mn.Vector3(0, 0, 0.3)),
                        ],
                        mn.Color4.blue(),
                    ),
                ]
                debug_lines.extend(path_point_render_lines)
                vdb.render_debug_lines(debug_lines)
                nav_pos_3d = [
                    np.array([xz[0], 0.0, xz[1]]) for xz in navmesh_offset
                ]
                cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
                cur_pos = [
                    np.array([xz[0], trans.translation[1], xz[2]])
                    for xz in cur_pos
                ]
                vdb.render_debug_circles(
                    [
                        (
                            pos,
                            0.25,
                            mn.Vector3(0, 1.0, 0),
                            mn.Color4.red()
                            if collision[-1]
                            else mn.Color4.magenta(),
                        )
                        for pos in cur_pos
                    ]
                )
                # render into the frames buffer
                vdb.get_observation(
                    look_at=robot_pos,
                    # from should be behind and above the robot
                    look_from=robot_pos
                    + trans.transform_vector(mn.Vector3(0, 1.5, 1.5)),
                    obs_cache=debug_video_frames,
                )
            time_since_debug_frame += 1.0 / vc._sim_freq
    except Exception:
        return 1.0

    vdb.make_debug_video(
        output_path="spot_nav_debug",
        fps=debug_framerate,
        obs_cache=debug_video_frames,
    )

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

    dist = float(np.linalg.norm(np.array((snapped - point))[[0, 2]]))
    return dist < nav_to_min_distance
