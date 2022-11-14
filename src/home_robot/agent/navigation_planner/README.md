# Navigation Planners

### Discrete Action Planner Usage

```
from home_robot.agent.navigation_planner.discrete_planner import DiscretePlanner

# See discrete_planner.py for argument info
discrete_planner = DiscretePlanner(
    turn_angle=30.0,
    collision_threshold=0.20,
    obs_dilation_selem_radius=3,
    goal_dilation_selem_radius=10,
    map_size_cm=4800,
    map_resolution=5,
    visualize=True,
    print_images=True,
    dump_location="datadump",
    exp_name="exp"
)

# See discrete_planner.py for argument info
# 
# outputs discrete action in
# class DiscreteActions(Enum):
#    stop = 0
#    move_forward = 1
#    turn_left = 2
#    turn_right = 3
#
discrete_action, _ = planner.plan(
    obstacle_map,
    goal_map,
    sensor_pose,
    found_goal
)
