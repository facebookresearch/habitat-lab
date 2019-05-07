from typing import Optional, List

import numpy as np

from habitat.core.simulator import ShortestPathPoint
from habitat.tasks.nav.nav_task import NavigationGoal, NavigationEpisode
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

GEODESIC_TO_EUCLID_RATIO_THRESHOLD = 1.1
NUMBER_RETRIES_PER_TARGET = 10
NEAR_DIST_LIMIT = 1
FAR_DIST_LIMIT = 30
ISLAND_RADIUS_LIMIT = 1.5


def _ratio_sample_rate(ratio):
    """

    :param ratio: geodesic distance ratio to Euclid distance
    :return: value between 0.008 and 0.144 for ration 1 and 1.1
    """
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(s, t, env, near_dist, far_dist):
    euclid_dist = np.power(np.power(s - t, 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:
        return False, 0
    d_separation = env._sim.geodesic_distance(s, t)
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < GEODESIC_TO_EUCLID_RATIO_THRESHOLD and np.random.rand() > _ratio_sample_rate(
        distances_ratio
    ):
        return False, 0
    if env._sim._sim.pathfinder.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    return True, d_separation


def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    target_position,
    shortest_paths=None,
    info=None,
) -> Optional[NavigationEpisode]:
    goals = [NavigationGoal(position=target_position)]
    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )


def get_action_shortest_path(
    env, source_position, source_rotation, goal_position, radius=0.05
) -> List[ShortestPathPoint]:
    backup_episodes = env.episodes
    if not env.episode_start_time:
        env.reset()
    env.episodes = [env.current_episode]
    env.reset()
    env.sim.set_agent_state(source_position, source_rotation)

    mode = "greedy"

    follower = ShortestPathFollower(env.sim, radius, False)
    follower.mode = mode

    shortest_path = []

    while not env.episode_over:
        action = follower.get_next_action(goal_position)
        state = env.sim.get_agent_state()
        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                state.rotation.components.tolist(),
                action.value,
            )
        )
        env.step(action.value)

    env.episodes = backup_episodes
    return [shortest_path]


def generate_pointnav_episode(env, num_episodes=-1, gen_shortest_path=True):
    episode = None
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        target_position = env._sim._sim.pathfinder.get_random_navigable_point()

        if (
            env._sim._sim.pathfinder.island_radius(target_position)
            < ISLAND_RADIUS_LIMIT
        ):
            continue

        for retry in range(NUMBER_RETRIES_PER_TARGET):
            source_position = (
                env._sim._sim.pathfinder.get_random_navigable_point()
            )

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                env,
                near_dist=NEAR_DIST_LIMIT,
                far_dist=FAR_DIST_LIMIT,
            )
            if is_compatible:
                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

                shortest_paths = None
                if gen_shortest_path:
                    shortest_paths = get_action_shortest_path(
                        env,
                        source_position=source_position,
                        source_rotation=source_rotation,
                        goal_position=target_position,
                    )

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=env._sim.config.SCENE,
                    start_position=source_position.tolist(),
                    start_rotation=source_rotation,
                    target_position=target_position.tolist(),
                    shortest_paths=shortest_paths,
                    info={"geodesic_distance": dist},
                )

                episode_count += 1
                yield episode
