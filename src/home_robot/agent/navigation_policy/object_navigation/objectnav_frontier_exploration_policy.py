import torch
import torch.nn as nn
import skimage.morphology

from agent.utils.morphology_utils import binary_dilation


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

    def forward(self, map_features, goal_category):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 8 + num_sem_categories, M, M)
            goal_category: semantic goal category

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size,)
        """
        goal_map, found_goal = self.reach_goal_if_in_map(map_features, goal_category)
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
        return goal_map, found_goal

    def reach_goal_if_in_map(self, map_features, goal_category):
        """If the goal category is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            category_map = map_features[e, goal_category[e] + 8, :, :]

            if (category_map == 1).sum() > 0:
                goal_map[e] = category_map == 1
                found_goal[e] = True

        return goal_map, found_goal

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
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )

        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                goal_map[e] = frontier_map[e]

        return goal_map
