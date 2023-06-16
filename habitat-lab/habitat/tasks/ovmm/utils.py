import habitat_sim


def find_closest_goal_index_within_distance(
    sim, goals, episode_id, use_all_viewpoints=False, max_dist=-1
):
    """Returns the index of the goal that has the closest viewpoint"""
    if use_all_viewpoints:
        goal_view_points = [
            v.agent_state.position for goal in goals for v in goal.view_points
        ]
        goal_indices = [
            i for i, goal in enumerate(goals) for v in goal.view_points
        ]
    else:
        goal_indices = list(range(len(goals)))
        goal_view_points = [
            g.view_points[0].agent_state.position for g in goals
        ]
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = sim.robot.base_pos
    path.requested_ends = goal_view_points
    sim.pathfinder.find_path(path)
    assert (
        path.closest_end_point_index != -1
    ), f"None of the goals are reachable from current position for episode {episode_id}"
    if max_dist != -1 and path.geodesic_distance > max_dist:
        return -1
    # RotDist to closest goal
    return goal_indices[path.closest_end_point_index]
