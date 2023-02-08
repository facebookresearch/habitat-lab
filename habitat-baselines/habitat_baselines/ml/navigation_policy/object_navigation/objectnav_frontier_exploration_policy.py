import skimage.morphology
import torch
import torch.nn as nn

from habitat_baselines.ml.utils.morphology_utils import binary_dilation


class ObjectNavFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(self):
        super().__init__()

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

    @property
    def goal_update_steps(self):
        return 1

    def forward(self, map_features, object_category=None, recep_category=None):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
            recep_category: receptacle goal category
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert object_category is not None or recep_category is not None
        if object_category is not None and recep_category is not None:
            # First check if object (small goal) and recep category are in the same cell of the map. if found, set it as a goal
            goal_map, found_goal = self.reach_goal_if_in_map(
                map_features, recep_category, small_goal_category=object_category,
            )
            # Then check if the recep category exists in the map. if found, set it as a goal
            goal_map, found_rec_goal = self.reach_goal_if_in_map(
                map_features, recep_category, reject_visited_regions=True, goal_map=goal_map, found_goal=found_goal
            )
            # Otherwise, set closest frontier as the goal
            goal_map = self.explore_otherwise(map_features, goal_map, found_rec_goal)
            return goal_map, found_goal

        else:
            # Here, the goal is specified by a single object or receptacle to navigate to with no additional constraints (eg. the given object can be on any receptacle)
            goal_category = object_category if object_category is not None else recep_category
            # if the goal is found, reach it
            goal_map, found_goal = self.reach_goal_if_in_map(
                map_features, goal_category
            )
            # otherwise, do frontier exploration
            goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
            return goal_map, found_goal

    def reach_goal_if_in_map(self, map_features, goal_category, small_goal_category=None, reject_visited_regions=False, goal_map=None, found_goal=None):
        """If the desired goal is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device
        if goal_map is None and found_goal is None:
            goal_map = torch.zeros((batch_size, height, width), device=device)
            found_goal_current = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            # crate a fresh map
            found_goal_current =  torch.clone(found_goal)
        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[e, goal_category[e] + 10, :, :]
                if small_goal_category is not None:
                    # additionally check if the category has the required small object on it
                    category_map = category_map * map_features[e, small_goal_category[e] + 10, :, :]
                if reject_visited_regions:
                    # remove the receptacles that the already been close to
                    category_map = category_map * (1 - map_features[e, 4, :, :])
                # if the desired category is found with required constraints, set goal for navigation
                if (category_map == 1).sum() > 0:
                    goal_map[e] = category_map == 1
                    found_goal_current[e] = True

        return goal_map, found_goal_current

    def explore_otherwise(self, map_features, goal_map, found_goal):
        """Explore closest unexplored region otherwise."""
        # Select unexplored area
        frontier_map = (map_features[:, [1], :, :] == 0).float()

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel)
            - frontier_map
        )

        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                goal_map[e] = frontier_map[e]

        return goal_map
