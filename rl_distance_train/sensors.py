import numpy as np
from typing import Any, Optional
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.nav import ImageGoalSensor

@registry.register_sensor(name="ImageGoalSensorV2")
class ImageGoalSensorV2(ImageGoalSensor):
    """
    ImageGoalSensorV2 extends ImageGoalSensor to support precomputed goal views ("views"),
    falling back to simulator-based generation if not present.
    """
    cls_uuid: str = "imagegoal_v2"

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal = episode.goals[0]
        if hasattr(goal, "views") and goal.views is not None:
            goal_views = goal.views
            # Deterministic sampling using episode_id
            seed = abs(hash(episode.episode_id)) % (2**32)
            rng = np.random.RandomState(seed)
            selected_view = rng.choice(goal_views)
            # If you want, handle disk loading here, or just return the object if already loaded
            return selected_view
        # Otherwise, call the parent method for classic sim-generated goal
        return super()._get_pointnav_episode_image_goal(episode)
