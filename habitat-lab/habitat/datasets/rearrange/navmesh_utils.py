from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

import habitat_sim
from habitat.articulated_agents.mobile_manipulator import MobileManipulator
from habitat.core.logging import logger
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.tasks.rearrange.utils import (
    general_sim_collision,
    get_angle_to_pos,
    rearrange_collision,
)
from habitat.tasks.utils import get_angle
from habitat_sim.physics import VelocityControl


def snap_point_is_occluded(
    target: mn.Vector3,
    snap_point: mn.Vector3,
    height: float,
    sim: habitat_sim.Simulator,
    granularity: float = 0.2,
    target_object_ids: Optional[List[int]] = None,
    ignore_object_ids: Optional[List[int]] = None,
) -> bool:
    """
    Uses raycasting to check whether a target is occluded given a navmesh snap point.

    :param target: The 3D position which should be unoccluded from the snap point.
    :param snap_point: The navmesh snap point under consideration.
    :param height: The height of the agent above the navmesh. Assumes the navmesh snap point is on the ground. Should be the maximum relative distance from navmesh ground to which a visibility check should indicate non-occlusion. The first check starts from this height. (E.g. agent_eyes_y - agent_base_y)
    :param sim: The Simulator instance.
    :param granularity: The distance between raycast samples. Finer granularity is more accurate, but more expensive.
    :param target_object_ids: An optional set of object ids which indicate the target. If one of these objects is hit before any non-ignored object, the test is successful.
    :param ignore_object_ids: An optional set of object ids which should be ignored in occlusion check.

    NOTE: If agent's eye height is known and only that height should be considered, provide eye height and granularity > height for fastest check.

    :return: whether or not the target is considered occluded from the snap_point.
    """

    if np.any(np.isnan(target)):
        raise ValueError(f"target point {target} is not valid.")

    if np.any(np.isnan(snap_point)):
        raise ValueError(f"snap_point {snap_point} is not valid.")

    # start from the top, assuming the agent's eyes are not at the bottom.
    cur_height = height
    while cur_height > 0:
        ray = habitat_sim.geo.Ray()
        ray.origin = snap_point + mn.Vector3(0, cur_height, 0)
        cur_height -= granularity
        ray.direction = target - ray.origin
        raycast_results = sim.cast_ray(ray)
        # distance of 1 is the displacement between the two points
        if (
            raycast_results.has_hits()
            and raycast_results.hits[0].ray_distance < 1
        ):
            for hit in raycast_results.hits:
                if hit.ray_distance > 1:
                    # exceeded the distance check without hitting an occlusion
                    return False

                if (
                    target_object_ids is not None
                    and hit.object_id in target_object_ids
                ):
                    # we hit an allowed object (i.e., the target object), so not occluded
                    return False
                elif (
                    ignore_object_ids is not None
                    and hit.object_id in ignore_object_ids
                ):
                    # we hit an ignored object, so continue the search
                    continue
                else:
                    # the ray hit a not-allowed object within distance threshold and is occluded at this height
                    break
        else:
            # ray hit nothing, so not occluded
            return False

    # we tried all heights and found no valid raycast, so the object is occluded
    return True


def unoccluded_navmesh_snap(
    pos: mn.Vector3,
    height: float,
    pathfinder: habitat_sim.nav.PathFinder,
    sim: habitat_sim.Simulator,
    target_object_ids: Optional[List[int]] = None,
    ignore_object_ids: Optional[List[int]] = None,
    island_id: int = -1,
    search_offset: float = 1.5,
    test_batch_size: int = 20,
    max_samples: int = 200,
    min_sample_dist: float = 0.5,
) -> Optional[mn.Vector3]:
    """
    Snap a point to the navmesh considering point visibility via raycasting.

    :param pos: The 3D position to snap.
    :param height: The height of the agent above the navmesh. Assumes the navmesh snap point is on the ground. Should be the maximum relative distance from navmesh ground to which a visibility check should indicate non-occlusion. The first check starts from this height. (E.g. agent_eyes_y - agent_base_y)
    :param pathfinder: The PathFinder defining the NavMesh to use.
    :param sim: The Simulator instance.
    :param target_object_ids: An optional set of object ids which indicate the target. If one of these objects is hit before any non-ignored object, the test is successful. For example, when pos is an object's COM, that object should not occlude the point.
    :param ignore_object_ids: An optional set of object ids which should be ignored in occlusion check. These objects should not stop the check. For example, the body and links of a robot.
    :param island_id: Optionally restrict the search to a single navmesh island. Default -1 is the full navmesh.
    :param search_offset: The additional radius to search for navmesh points around the target position. Added to the minimum distance from pos to navmesh.
    :param test_batch_size: The number of sample navmesh points to consider when testing for occlusion.
    :param max_samples: The maximum number of attempts to sample navmesh points for the test batch.
    :param min_sample_dist: The minimum allowed L2 distance between samples in the test batch.

    NOTE: this function is based on sampling and does not guarantee the closest point.

    :return: An approximation of the closest unoccluded snap point to pos or None if an unoccluded point could not be found.
    """

    # first try the closest snap point
    snap_point = pathfinder.snap_point(pos, island_id)

    is_occluded = np.isnan(snap_point[0]) or snap_point_is_occluded(
        target=pos,
        snap_point=snap_point,
        height=height,
        sim=sim,
        target_object_ids=target_object_ids,
        ignore_object_ids=ignore_object_ids,
    )

    # now sample and try different snap options
    if is_occluded:
        # distance to closest snap point is the absolute minimum
        min_radius = (snap_point - pos).length()
        # expand the search radius
        search_radius = min_radius + search_offset

        # gather a test batch
        test_batch: List[Tuple[mn.Vector3, float]] = []
        sample_count = 0
        while len(test_batch) < test_batch_size and sample_count < max_samples:
            sample = pathfinder.get_random_navigable_point_near(
                circle_center=pos, radius=search_radius, island_index=island_id
            )
            if not np.isnan(sample[0]):
                reject = False
                for batch_sample in test_batch:
                    if (
                        np.linalg.norm(sample - batch_sample[0])
                        < min_sample_dist
                    ):
                        reject = True
                        break
                if not reject:
                    test_batch.append(
                        (sample, float(np.linalg.norm(sample - pos)))
                    )
            sample_count += 1

        # sort the test batch points by distance to the target
        test_batch.sort(key=lambda s: s[1])

        # find the closest unoccluded point in the test batch
        for batch_sample in test_batch:
            if not snap_point_is_occluded(
                pos,
                batch_sample[0],
                height,
                sim,
                target_object_ids=target_object_ids,
                ignore_object_ids=ignore_object_ids,
            ):
                return batch_sample[0]

        # unable to find an unoccluded point
        return None

    # the true closest snap point is unoccluded
    return snap_point


def embodied_unoccluded_navmesh_snap(
    target_position: mn.Vector3,
    height: float,
    sim: habitat_sim.Simulator,
    pathfinder: habitat_sim.nav.PathFinder = None,
    target_object_ids: Optional[List[int]] = None,
    ignore_object_ids: Optional[List[int]] = None,
    ignore_object_collision_ids: Optional[List[int]] = None,
    island_id: int = -1,
    search_offset: float = 1.5,
    test_batch_size: int = 20,
    max_samples: int = 200,
    min_sample_dist: float = 0.5,
    embodiment_heuristic_offsets: Optional[List[mn.Vector2]] = None,
    agent_embodiment: Optional[MobileManipulator] = None,
    orientation_noise: float = 0,
    max_orientation_samples: int = 5,
    data_out: Dict[Any, Any] = None,
) -> Tuple[mn.Vector3, float, bool]:
    """
    Snap a robot embodiment close to a target point considering embodied constraints via the navmesh and raycasting for point visibility.

    :param target_position: The 3D target position to snap.
    :param height: The height of the agent above the navmesh. Assumes the navmesh snap point is on the ground. Should be the maximum relative distance from navmesh ground to which a visibility check should indicate non-occlusion. The first check starts from this height. (E.g. agent_eyes_y - agent_base_y)
    :param sim: The RearrangeSimulator or Simulator instance. This choice will dictate the collision detection routine.
    :param pathfinder: The PathFinder defining the NavMesh to use.
    :param target_object_ids: An optional set of object ids which indicate the target. If one of these objects is hit before any non-ignored object, the test is successful. For example, when pos is an object's COM, that object should not occlude the point.
    :param ignore_object_ids: An optional set of object ids which should be ignored in occlusion check. These objects should not stop the check. For example, the body and links of a robot.
    :param ignore_object_collision_ids: An optional set of object ids which should be ignored in collision check. These objects should not stop the check. For example, the body and links of a robot.
    :param island_id: Optionally restrict the search to a single navmesh island. Default -1 is the full navmesh.
    :param search_offset: The additional radius to search for navmesh points around the target position. Added to the minimum distance from pos to navmesh.
    :param test_batch_size: The number of sample navmesh points to consider when testing for occlusion.
    :param max_samples: The maximum number of attempts to sample navmesh points for the test batch.
    :param min_sample_dist: The minimum allowed L2 distance between samples in the test batch.
    :param embodiment_heuristic_offsets: A set of 2D offsets describing navmesh cylinder center points forming a proxy for agent embodiment. Assumes x-forward, y to the side and 3D height fixed to navmesh. If provided, this proxy embodiment will be used for collision checking. If provided with an agent_embodiment, will be used instead of the MobileManipulatorParams.navmesh_offsets
    :param agent_embodiment: The MobileManipulator to be used for collision checking if provided.
    :param orientation_noise: Standard deviation of the gaussian used to sample orientation noise. If 0, states always face the target point. Noise is applied delta to this "target facing" orientation.
    :param max_orientation_samples: The number of orientation noise samples to try for each candidate point.
    :param data_out: Optionally provide a dictionary which can be filled with arbitrary detail data for external debugging and visualization.

    NOTE: this function is based on sampling and does not guarantee the closest point.

    :return: A Tuple containing: 1) An approximation of the closest unoccluded snap point to pos or None if an unoccluded point could not be found, 2) the sampled orientation if found or None, 3) a boolean success flag.
    """

    assert height > 0
    assert search_offset > 0
    assert test_batch_size > 0
    assert max_samples > 0
    assert orientation_noise >= 0

    if np.any(np.isnan(target_position)):
        raise ValueError(f"target_position {target_position} is not valid.")

    if pathfinder is None:
        pathfinder = sim.pathfinder

    assert pathfinder.is_loaded

    # when an agent_embodiment is provided, use its navmesh_offsets unless overridden by input
    if embodiment_heuristic_offsets is None and agent_embodiment is not None:
        embodiment_heuristic_offsets = agent_embodiment.params.navmesh_offsets

    # set the search radius
    search_radius = search_offset
    # try the closest snap point to find expected distance
    snap_point = pathfinder.snap_point(target_position, island_id)
    if not np.any(np.isnan(snap_point)):
        # distance to closest snap point is the absolute minimum radius
        min_radius = (snap_point - target_position).length()
        # expand the search radius
        search_radius = min_radius + search_offset

    # gather a test batch
    test_batch: List[Tuple[mn.Vector3, float]] = []
    sample_count = 0
    while len(test_batch) < test_batch_size and sample_count < max_samples:
        sample = pathfinder.get_random_navigable_point_near(
            circle_center=target_position,
            radius=search_radius,
            island_index=island_id,
        )
        # validate the sample before caching
        if not np.any(np.isnan(sample)):
            reject = False
            for batch_sample in test_batch:
                if np.linalg.norm(sample - batch_sample[0]) < min_sample_dist:
                    reject = True
                    break
            if not reject:
                test_batch.append(
                    (sample, float(np.linalg.norm(sample - target_position)))
                )
        sample_count += 1

    # sort the test batch points by distance to the target
    test_batch.sort(key=lambda s: s[1])

    # find the closest unoccluded point in the test batch
    for batch_sample in test_batch:
        if not snap_point_is_occluded(
            target_position,
            batch_sample[0],
            height,
            sim,
            target_object_ids=target_object_ids,
            ignore_object_ids=ignore_object_ids,
        ):
            facing_target_angle = get_angle_to_pos(
                np.array(target_position - batch_sample[0])
            )

            if (
                embodiment_heuristic_offsets is None
                and agent_embodiment is None
            ):
                # No embodiment for collision detection, so return closest unoccluded point
                return batch_sample[0], facing_target_angle, True

            # get orientation noise offset
            orientation_noise_samples = []
            if orientation_noise > 0 and max_orientation_samples > 0:
                orientation_noise_samples = [
                    np.random.normal(0.0, orientation_noise)
                    for _ in range(max_orientation_samples)
                ]
            # last one is always no-noise to check forward-facing
            orientation_noise_samples.append(0)
            for orientation_noise_sample in orientation_noise_samples:
                desired_angle = facing_target_angle + orientation_noise_sample
                if embodiment_heuristic_offsets is not None:
                    # local 2d point rotation
                    rotation_2d = mn.Matrix3.rotation(-mn.Rad(desired_angle))
                    transformed_offsets_2d = [
                        rotation_2d.transform_vector(xz)
                        for xz in embodiment_heuristic_offsets
                    ]

                    # translation to global 3D points at navmesh height
                    offsets_3d = [
                        np.array(
                            [
                                transformed_offset_2d[0],
                                0,
                                transformed_offset_2d[1],
                            ]
                        )
                        + batch_sample[0]
                        for transformed_offset_2d in transformed_offsets_2d
                    ]

                    if data_out is not None:
                        data_out["offsets_3d"] = offsets_3d

                    # check for offset navigability
                    is_collision = False
                    for offset_point in offsets_3d:
                        if not (
                            sim.pathfinder.is_navigable(offset_point)
                            and (
                                island_id == -1
                                or sim.pathfinder.get_island(offset_point)
                                == island_id
                            )
                        ):
                            is_collision = True
                            break

                    # if this sample is invalid, try the next
                    if is_collision:
                        continue

                if agent_embodiment is not None:
                    # contact testing with collision shapes
                    start_position = agent_embodiment.base_pos
                    start_rotation = agent_embodiment.base_rot

                    agent_embodiment.base_pos = batch_sample[0]
                    agent_embodiment.base_rot = desired_angle

                    details = None
                    sim.perform_discrete_collision_detection()
                    # Make sure the robot is not colliding with anything in this state.
                    if sim.__class__.__name__ == "RearrangeSim":
                        _, details = rearrange_collision(
                            sim,
                            False,
                            ignore_object_ids=ignore_object_collision_ids,
                            ignore_base=False,
                        )
                    else:
                        _, details = general_sim_collision(
                            sim,
                            agent_embodiment,
                            ignore_object_ids=ignore_object_collision_ids,
                        )

                    # reset agent state
                    agent_embodiment.base_pos = start_position
                    agent_embodiment.base_rot = start_rotation

                    # Only care about collisions between the robot and scene.
                    is_feasible_state = details.robot_scene_colls == 0
                    if not is_feasible_state:
                        continue

                # if we made it here, all tests passed and we found a valid placement state
                return batch_sample[0], desired_angle, True

    # unable to find a valid navmesh point within constraints
    return None, None, False


def is_collision(
    pathfinder: habitat_sim.nav.PathFinder,
    trans: mn.Matrix4,
    navmesh_offset: List[Tuple[float, float]],
    island_idx: int,
) -> bool:
    """
    Checks the given transform and navmesh offset points for navigability on the provided navmesh island. Returns True if any point is non-navigable.

    :param pathfinder: The PathFinder instance defining the NavMesh.
    :param trans: The current agent transformation matrix.
    :param navmesh_offset: A list of 2D navmesh offset points to check.
    :param largest_island_idx: The index of the island to query. -1 is the entire navmesh.
    """
    nav_pos_3d = [np.array([xz[0], 0.0, xz[1]]) for xz in navmesh_offset]
    cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
    cur_pos = [
        np.array([xz[0], trans.translation[1], xz[2]]) for xz in cur_pos
    ]

    for pos in cur_pos:
        # Return True if the point is not navigable on the configured island
        # TODO: pathfinder.is_navigable does not support island specification, so duplicating functionality for now
        largest_island_snap_point = pathfinder.snap_point(
            pos, island_index=island_idx
        )
        vertical_dist = abs(largest_island_snap_point[1] - pos[1])
        if vertical_dist > 0.5:
            return True
        horizontal_dist = np.linalg.norm(
            np.array(largest_island_snap_point)[[0, 2]] - pos[[0, 2]]
        )
        if horizontal_dist > 0.01:
            return True

    return False


def compute_turn(
    target: np.ndarray, turn_speed: float, robot_forward: np.ndarray
) -> List[float]:
    """
    Computes the constant speed angular velocity about the Y axis to turn the 2D robot_forward vector toward the provided 2D target direction in global coordinates.

    :param target: The 2D global target vector in XZ.
    :param turn_speed: The desired constant turn speed.
    :param robot_forward: The global 2D robot forward vector in XZ.
    """
    is_left = np.cross(robot_forward, target) > 0
    if is_left:
        vel = [0, -turn_speed]
    else:
        vel = [0, turn_speed]
    return vel


class SimpleVelocityControlEnv:
    """
    A simple environment to control the velocity of the robot.
    Assumes x-forward in robot local space.
    """

    def __init__(self, integration_frequency: float = 60.0):
        """
        Initialize the internal VelocityControl object.

        :param integration_frequency: The frequency of integration. Number of integration steps in a second. Integration step size = 1.0/integration_frequency.
        """

        # the velocity control
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._integration_frequency = integration_frequency

    def act(self, trans: mn.Matrix4, vel: List[float]) -> mn.Matrix4:
        """
        Integrate the current linear and angular velocity and return the new transform.

        :param trans: The current agent transformation matrix.
        :param vel: 2D list of linear (forward) and angular (about Y) velocity.

        :return: The updated agent transformation matrix.
        """

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
        # Get the target rigid state based on the simulation frequency
        target_rigid_state = self.vel_control.integrate_transform(
            1.0 / self._integration_frequency, rigid_state
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


def record_robot_nav_debug_image(
    curr_path_points: List[mn.Vector3],
    robot_transformation: mn.Matrix4,
    robot_navmesh_offsets: List[Tuple[float, float]],
    robot_navmesh_radius: float,
    in_collision: bool,
    dbv: DebugVisualizer,
    obs_cache: List[Any],
) -> None:
    """
    Render a single frame 3rd person view of the robot embodiment approximation following a path with DebugVizualizer and cache it in obs_cache.

    :param curr_path_points: List of current path points.
    :param robot_transformation: Current transformation of the robot.
    :param robot_navmesh_offsets: Robot embodiment approximation. List of 2D points XZ in robot local space.
    :param robot_navmesh_radius: The radius of each point approximating the robot embodiment.
    :param in_collision: Whether or not the robot is in collision with the environment. If so, embodiment is rendered red.
    :param dbv: The DebugVisualizer instance.
    :param obs_cache: The observation cache for later video rendering.
    """

    # render the provided path
    path_point_render_lines = []
    for i in range(len(curr_path_points)):
        if i > 0:
            path_point_render_lines.append(
                (
                    [curr_path_points[i - 1], curr_path_points[i]],
                    mn.Color4.cyan(),
                )
            )
    dbv.render_debug_lines(debug_lines=path_point_render_lines)

    # draw the local coordinate axis of the robot
    dbv.render_debug_frame(
        axis_length=0.3, transformation=robot_transformation
    )

    # render the robot embodiment
    nav_pos_3d = [
        np.array([xz[0], 0.0, xz[1]]) for xz in robot_navmesh_offsets
    ]
    cur_pos = [robot_transformation.transform_point(xyz) for xyz in nav_pos_3d]
    cur_pos = [
        np.array([xz[0], robot_transformation.translation[1], xz[2]])
        for xz in cur_pos
    ]
    dbv.render_debug_circles(
        [
            (
                pos,
                robot_navmesh_radius,
                mn.Vector3(0, 1.0, 0),
                mn.Color4.red() if in_collision else mn.Color4.magenta(),
            )
            for pos in cur_pos
        ]
    )

    # render 3rd person viewer into the observation cache
    robot_position = robot_transformation.translation
    obs_cache.append(
        dbv.get_observation(
            look_at=robot_position,
            # 3rd person viewpoint from behind and above the robot
            look_from=robot_position
            + robot_transformation.transform_vector(mn.Vector3(0, 1.5, 1.5)),
        )
    )


def path_is_navigable_given_robot(
    sim: habitat_sim.Simulator,
    start_pos: mn.Vector3,
    goal_pos: mn.Vector3,
    robot_navmesh_offsets: List[Tuple[float, float]],
    collision_rate_threshold: float,
    selected_island: int = -1,
    angle_threshold: float = 0.05,
    angular_speed: float = 1.0,
    distance_threshold: float = 0.25,
    linear_speed: float = 1.0,
    dbv: Optional[DebugVisualizer] = None,
    render_debug_video: bool = False,
) -> bool:
    """
    Compute the ratio of time-steps for which there were collisions detected while the robot navigated from start_pos to goal_pos given the configuration of the sim navmesh.

    :param sim: Habitat Simulator instance.
    :param start_pos: Initial translation of the robot's transform. The start of the navigation path.
    :param goal_pos: Target translation of the robot's transform. The end of the navigation path.
    :param robot_navmesh_offsets: The list of 2D points XZ in robot local space which will be used represent the robot's shape. Used to query the navmesh for navigability as a collision heuristic.
    :param collision_rate_threshold: The acceptable ratio of colliding to non-colliding steps in the navigation path. Collision is computed with a heuristic, so should be non-zero.
    :param selected_island: The navmesh island to which queries should be constrained. -1 denotes the full navmesh.
    :param angle_threshold: The error threshold in radians over which the robot should turn before moving straight.
    :param angular_speed: The constant angular speed for turning (radians/sec)
    :param distance_threshold: The euclidean distance between the robot and the target within which navigation is considered successful and the function returns.
    :param linear_speed: The constant linear speed for translation (meters/sec).
    :param dbv: An optional DebugVisualizer if rendering and video export are desired.
    :param render_debug_video: Whether or not to render and export a visualization of the navigation. If True, requires a DebugVisualizer instance.

    :return: Whether or not the ratio of time-steps where collisions were detected is within the provided threshold.
    """
    logger.info(
        "Checking robot navigability between target object start and goal:"
    )

    snapped_start_pos = sim.pathfinder.snap_point(start_pos, selected_island)
    snapped_goal_pos = sim.pathfinder.snap_point(goal_pos, selected_island)

    logger.info(
        f"     - start_pos to snapped_start_pos distance = {(snapped_start_pos-start_pos).length()}"
    )
    logger.info(
        f"     - goal_pos to snapped_goal_pos distance = {(snapped_goal_pos-goal_pos).length()}"
    )

    if render_debug_video:
        assert dbv is not None, "Need a dbv for visual debugging."
        sim.navmesh_visualization = True

    # Create a new pathfinder with slightly stricter radius to provide nav buffer from collision
    pf = habitat_sim.nav.PathFinder()
    modified_settings = sim.pathfinder.nav_mesh_settings
    robot_navmesh_radius = modified_settings.agent_radius
    modified_settings.agent_radius += 0.05
    assert sim.recompute_navmesh(
        pf, modified_settings
    ), "failed to recompute navmesh"
    # Init the shortest path
    path = habitat_sim.ShortestPath()
    # Set the locations
    path.requested_start = snapped_start_pos
    path.requested_end = snapped_goal_pos
    # Find the path
    found_path = pf.find_path(path)
    if not found_path:
        logger.info(
            f"     - cannot find path between start_pos({start_pos}) and goal_pos({goal_pos})."
        )
        return False
    curr_path_points = path.points
    # Set the initial position
    p0 = mn.Vector3(curr_path_points[0])
    p1 = mn.Vector3(curr_path_points[1])
    p1[1] = curr_path_points[0][1]
    trans: mn.Matrix4 = mn.Matrix4.look_at(
        eye=p0, target=p1, up=mn.Vector3(0.0, 1.0, 0.0)
    )

    # Get the robot position
    robot_pos = np.array(trans.translation)
    # Get the navigation target
    final_nav_targ = np.array(curr_path_points[-1])
    obj_targ_pos = np.array(curr_path_points[-1])
    # the velocity control
    vc = SimpleVelocityControlEnv()
    forward = np.array([1.0, 0, 0])

    at_goal = False

    collision = []
    debug_video_frames: List[Any] = []
    debug_framerate = 30
    time_since_debug_frame = 9999.0

    while not at_goal:
        # Find the path
        path.requested_start = robot_pos
        path.requested_end = snapped_goal_pos
        pf.find_path(path)
        curr_path_points = path.points
        cur_nav_targ = np.array(curr_path_points[1])
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
        at_goal = bool(
            dist_to_final_nav_targ < distance_threshold
            and angle_to_obj < angle_threshold
        )

        if not at_goal:
            if dist_to_final_nav_targ < distance_threshold:
                # Do not want to look at the object to reduce collision
                vel = [0.0, 0.0]
                at_goal = True
            elif angle_to_target < angle_threshold:
                # Move towards the target
                vel = [linear_speed, 0.0]
            else:
                # Look at the target waypoint.
                vel = compute_turn(rel_targ, angular_speed, robot_forward)
        else:
            vel = [0.0, 0.0]
        trans = vc.act(trans, vel)
        robot_pos = trans.translation
        collision.append(
            is_collision(
                sim.pathfinder, trans, robot_navmesh_offsets, selected_island
            )
        )
        if (
            render_debug_video
            and time_since_debug_frame > 1.0 / debug_framerate
        ):
            time_since_debug_frame = 0
            record_robot_nav_debug_image(
                curr_path_points=curr_path_points,
                robot_transformation=trans,
                robot_navmesh_offsets=robot_navmesh_offsets,
                robot_navmesh_radius=robot_navmesh_radius,
                in_collision=collision[-1],
                dbv=dbv,
                obs_cache=debug_video_frames,
            )
        time_since_debug_frame += 1.0 / vc._integration_frequency

    collision_rate = np.average(collision)

    if render_debug_video:
        dbv.make_debug_video(
            output_path="spot_nav_debug",
            prefix=f"{collision_rate}",
            fps=debug_framerate,
            obs_cache=debug_video_frames,
        )

    logger.info(f"  collision rate {collision_rate}")
    if collision_rate > collision_rate_threshold:
        return False
    return True


def is_accessible(
    sim: habitat_sim.Simulator,
    point: mn.Vector3,
    height: float,
    nav_to_min_distance: float,
    nav_island: int = -1,
    target_object_ids: Optional[List[int]] = None,
) -> bool:
    """
    Return True if the point is within a threshold distance (in XZ plane) of the nearest unoccluded navigable point on the selected island.

    :param sim: Habitat Simulator instance.
    :param point: The query point.
    :param height: The height of the agent. Given navmesh snap point is grounded, the maximum height from which a visibility check should indicate non-occlusion. First check starts from this height.
    :param nav_to_min_distance: Minimum distance threshold. -1 opts out of the test and returns True (i.e. no minimum distance).
    :param nav_island: The NavMesh island on which to check accessibility. Default -1 is the full NavMesh.
    :param target_object_id: An optional set of object ids which should be ignored in occlusion check. For example, when checking accessibility of an object's COM, that object should not occlude.

    :return: Whether or not the point is accessible.
    """
    if nav_to_min_distance == -1:
        return True

    snapped = unoccluded_navmesh_snap(
        pos=point,
        height=height,
        pathfinder=sim.pathfinder,
        sim=sim,
        target_object_ids=target_object_ids,
        island_id=nav_island,
        search_offset=nav_to_min_distance,
    )

    if snapped is None:
        return False

    horizontal_dist = float(
        np.linalg.norm(np.array((snapped - point))[[0, 2]])
    )
    return horizontal_dist < nav_to_min_distance


def is_outdoor(
    pathfinder: habitat_sim.nav.PathFinder,
    sim: habitat_sim.Simulator,
    island_ix: int,
    num_samples: int = 100,
    indoor_ratio_threshold: float = 0.95,
    min_sample_dist: Optional[float] = None,
    max_sample_attempts: int = 200,
) -> bool:
    """
    Heuristic to check if the specified NavMesh island is outdoor or indoor.

    :param pathfinder: The NavMesh to check.
    :param sim: The Simulator instance.
    :param island_ix: The index of the island to check. -1 for all islands.
    :param num_samples: The number of samples to take.
    :param indoor_ratio_threshold: The percentage of samples classified as indoor necessary to pass the test.
    :param min_sample_dist: (optional) The minimum distance between samples. Default is no minimum distance.
    :param max_sample_attempts: The maximum number of sample to attempt to satisfy minimum distance.

    Assumptions:
     1. The scene must have ceiling collision geometry covering all indoor areas.
     2. Indoor and outdoor spaces are separate navmeshes. Mixed interior/exterior navmeshes may be classified incorrectly and non-deterministically as the heuristic is based on sampling and thresholding.
    """

    assert pathfinder.is_loaded, "PathFinder is not loaded."

    # 1. Sample the navmesh (pathfinder) island (island_ix) until num_samples achieved with with pairwise min_sample_dist or max_sample_attempts.
    num_tries = 0
    nav_samples: List[np.ndarray] = []
    while len(nav_samples) < num_samples and num_tries < max_sample_attempts:
        nav_sample = pathfinder.get_random_navigable_point(
            island_index=island_ix
        )
        if np.any(np.isnan(nav_sample)):
            continue
        if min_sample_dist is not None:
            too_close = False
            for existing_sample in nav_samples:
                sample_distance = np.linalg.norm(nav_sample - existing_sample)
                if sample_distance < min_sample_dist:
                    too_close = True
                    break
            if too_close:
                continue
        nav_samples.append(nav_sample)

    # 2. For each sample, raycast in +Y direction.
    #    - Any hit points classify the sample as indoor, otherwise outdoor.
    up = mn.Vector3(0, 1.0, 0)
    ray_results = [
        sim.cast_ray(habitat_sim.geo.Ray(nav_sample, up))
        for nav_sample in nav_samples
    ]
    num_indoor_samples = sum([results.has_hits() for results in ray_results])

    # 3. Compute percentage of indoor samples and compare against indoor_ratio_threshold
    indoor_ratio = float(num_indoor_samples) / len(nav_samples)
    return indoor_ratio <= indoor_ratio_threshold


def get_largest_island_index(
    pathfinder: habitat_sim.nav.PathFinder,
    sim: habitat_sim.Simulator,
    allow_outdoor: bool = True,
) -> int:
    """
    Get the index of the largest NavMesh island.
    Optionally exclude outdoor islands.

    NOTE: outdoor heuristic may need to be tuned, but parameters are default here.

    If no islands exist satisfying the indoor constraints, then the entire navmesh -1 is returned.
    """

    assert pathfinder.is_loaded, "PathFinder is not loaded."

    # get list of (island_index,area) tuples
    island_areas = [
        (island_ix, pathfinder.island_area(island_index=island_ix))
        for island_ix in range(pathfinder.num_islands)
    ]
    # sort by area, descending
    island_areas.sort(reverse=True, key=lambda x: x[1])

    if not allow_outdoor:
        # classify indoor vs outdoor
        island_outdoor_classifications = [
            is_outdoor(pathfinder, sim, island_info[0])
            for island_info in island_areas
        ]
        if False not in island_outdoor_classifications:
            return -1
        # select the largest outdoor island
        largest_indoor_island = island_areas[
            island_outdoor_classifications.index(False)
        ][0]
        return largest_indoor_island

    return island_areas[0][0]
