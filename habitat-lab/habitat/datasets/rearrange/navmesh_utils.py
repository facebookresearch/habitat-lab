from typing import List

import magnum as mn
import numpy as np

import habitat_sim


def is_collision(sim, navmesh_offset) -> bool:
    """
    The function checks if the agent collides with the object
    """

    trans = sim.agents[0].scene_node.transformation
    nav_pos_3d = [np.array([xz[0], 0.0, xz[1]]) for xz in navmesh_offset]
    cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
    is_navigable = [sim.pathfinder.is_navigable(pos) for pos in cur_pos]

    # No collision for each navmesh circles
    if sum(is_navigable) == len(navmesh_offset):
        return False

    return True


def compute_turn(rel, turn_vel, robot_forward):
    is_left = np.cross(robot_forward, rel) > 0
    if is_left:
        vel = [0, -turn_vel]
    else:
        vel = [0, turn_vel]
    return vel


from habitat_sim.physics import VelocityControl


class velocity_control:
    def __init__(self):
        # the velocity control
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

    def act(self, sim, vel):
        ori_height = sim.agents[0].scene_node.translation[1]
        linear_velocity = vel[0]
        angular_velocity = vel[1]
        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3(
            [linear_velocity, 0.0, 0.0]
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0]
        )

        trans = sim.agents[0].scene_node.transformation
        current_rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        # manually integrate the current_rigid_state
        goal_rigid_state = self.vel_control.integrate_transform(
            1.0 / 60.0, current_rigid_state
        )

        goal_rigid_state.translation[1] = ori_height
        sim.agents[0].scene_node.translation = goal_rigid_state.translation
        sim.agents[0].scene_node.rotation = goal_rigid_state.rotation
        return sim


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

    We first get the points in the path. Then we do
    interpolation to get the points given the distance threshold.
    Then, for each point, we rotate the robot to see if there is a
    single pose that can fit the robot into the space given the navmesh.
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
    points = path.points
    # Set the initial position
    sim.agents[0].scene_node.translation = points[0]

    # the velocity control
    vc = velocity_control()

    # Number of collision
    collision = []
    for i in range(len(points) - 1):
        # Get the current location
        cur_pos = points[i]
        sim.agents[0].scene_node.translation = cur_pos
        # Get the next location
        next_pos = points[i + 1]

        # Failure detection counter
        while_counter = 0
        angle = float("inf")
        while abs(angle) > angle_threshold:
            # Compute the robot facing orientation
            rel_pos = (next_pos - cur_pos)[[0, 2]]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(
                sim.agents[0].scene_node.transformation.transform_vector(
                    forward
                )
            )
            robot_forward = robot_forward[[0, 2]]
            angle = np.cross(robot_forward, rel_pos)
            vel = compute_turn(rel_pos, angular_velocity, robot_forward)
            sim = vc.act(sim, vel)
            cur_pos = sim.agents[0].scene_node.translation
            collision.append(is_collision(sim, navmesh_offset))
            while_counter += 1
            if while_counter >= 1000:
                return 1.0

        # Failure detection counter
        while_counter = 0
        dis = np.linalg.norm((next_pos - cur_pos)[[0, 2]])
        while abs(dis) > distance_threshold:
            vel = [linear_velocity, 0]
            sim = vc.act(sim, vel)
            cur_pos = sim.agents[0].scene_node.translation
            dis = np.linalg.norm((next_pos - cur_pos)[[0, 2]])
            collision.append(is_collision(sim, navmesh_offset))
            while_counter += 1
            if while_counter >= 1000:
                return 1.0

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
    island_idx: int = sim.pathfinder.get_island(snapped)
    dist = float(np.linalg.norm(np.array((snapped - point))[[0, 2]]))
    return (
        dist < nav_to_min_distance
        and island_idx == sim.navmesh_classification_results["active_island"]
    )


def compute_navmesh_island_classifications(
    sim: habitat_sim.Simulator, active_indoor_threshold=0.85, debug=False
):
    """
    Classify navmeshes as outdoor or indoor and find the largest indoor island.
    active_indoor_threshold is acceptacle indoor|outdoor ration for an active island (for example to allow some islands with a small porch or skylight)
    """
    if not sim.pathfinder.is_loaded:
        sim.navmesh_classification_results = None
        print("No NavMesh loaded to visualize.")
        return

    sim.navmesh_classification_results = {}

    sim.navmesh_classification_results["active_island"] = -1
    sim.navmesh_classification_results[
        "active_indoor_threshold"
    ] = active_indoor_threshold
    active_island_size = 0
    number_of_indoor = 0
    sim.navmesh_classification_results["island_info"] = {}
    sim.indoor_islands = []

    for island_ix in range(sim.pathfinder.num_islands):
        sim.navmesh_classification_results["island_info"][island_ix] = {}
        sim.navmesh_classification_results["island_info"][island_ix][
            "indoor"
        ] = island_indoor_metric(sim=sim, island_ix=island_ix)
        if (
            sim.navmesh_classification_results["island_info"][island_ix][
                "indoor"
            ]
            > active_indoor_threshold
        ):
            number_of_indoor += 1
            sim.indoor_islands.append(island_ix)
        island_size = sim.pathfinder.island_area(island_ix)
        if (
            active_island_size < island_size
            and sim.navmesh_classification_results["island_info"][island_ix][
                "indoor"
            ]
            > active_indoor_threshold
        ):
            active_island_size = island_size
            sim.navmesh_classification_results["active_island"] = island_ix
    if debug:
        print(
            f"Found active island {sim.navmesh_classification_results['active_island']} with area {active_island_size}."
        )
        print(
            f"     Found {number_of_indoor} indoor islands out of {sim.pathfinder.num_islands} total."
        )
    for island_ix in range(sim.pathfinder.num_islands):
        island_info = sim.navmesh_classification_results["island_info"][
            island_ix
        ]
        info_str = f"    {island_ix}: indoor ratio = {island_info['indoor']}, area = {sim.pathfinder.island_area(island_ix)}"
        if sim.navmesh_classification_results["active_island"] == island_ix:
            info_str += "  -- active--"
    if debug:
        print(info_str)


def island_indoor_metric(
    sim: habitat_sim.Simulator,
    island_ix: int,
    num_samples=100,
    jitter_dist=0.1,
    max_tries=1000,
) -> float:
    """
    Compute a heuristic for ratio of an island inside vs. outside based on checking whether there is a roof over a set of sampled navmesh points.
    """

    assert sim.pathfinder.is_loaded
    assert sim.pathfinder.num_islands > island_ix

    # collect jittered samples
    samples: List[np.ndarray] = []
    for _sample_ix in range(max_tries):
        new_sample = sim.pathfinder.get_random_navigable_point(
            island_index=island_ix
        )
        too_close = False
        for prev_sample in samples:
            dist_to = np.linalg.norm(prev_sample - new_sample)
            if dist_to < jitter_dist:
                too_close = True
                break
        if not too_close:
            samples.append(new_sample)
        if len(samples) >= num_samples:
            break

    # classify samples
    indoor_count = 0
    for sample in samples:
        raycast_results = sim.cast_ray(
            habitat_sim.geo.Ray(sample, mn.Vector3(0, 1, 0))
        )
        if raycast_results.has_hits():
            # assume any hit indicates "indoor"
            indoor_count += 1

    # return the ration of indoor to outdoor as the metric
    return indoor_count / len(samples)
