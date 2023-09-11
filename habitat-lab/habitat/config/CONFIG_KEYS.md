# Configuration Keys

## Dataset
Configuration for the dataset of the task.
A dataset consists of episodes
(a start configuration for a task within a scene) and a scene dataset
(with all the assets needed to instantiate the task).
For a list of all the available datasets, see [this page](../../../DATASETS.md).

| Key | Description |
| --- | --- |
| habitat.dataset.type |  The key for the dataset class that will be used. Examples of such keys are `PointNav-v1`, `ObjectNav-v1`, `InstanceImageNav-v1` or `RearrangeDataset-v0`. Different datasets have different properties so you should use the dataset that fits your task. |
| habitat.dataset.scene_dir | The path to the directory containing the scenes that will be used. You should put all your scenes in the same folder (example `data/scene_datasets`) to avoid having to change it. |
|habitat.dataset.data_path | The path to the episode dataset. Episodes need to be compatible with the `type` argument (so they will load properly) and only use scenes that are present in the `scenes_dir`.|
|habitat.dataset.split | `data_path` can have a `split` in the path. For example: "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz" the value in "{split}" will be replaced by the value of the `split` argument. This allows to easily swap between training, validation and test episodes by only changing the split argument. |

## Task
The definition of the task in Habitat.
There are many different Tasks determined by the `habitat.task.type` config:
 - Point navigation : `Nav-0` The agent needs to navigate to a geometric goal position.
 - Image navigation : `Nav-0` The agent needs to navigate to an object matching the type shown in a goal image.
 - Instance image navigation:`InstanceImageNav-v1` The agent needs to navigate to an object matching the type shown in a goal image that is sampled from the current scene.
 - Object navigation : `ObjectNav-v1` The agent needs to navigate to a specific type of object in the scene indicated by a categorical value.
 - Rearrangement close drawer: `RearrangeCloseDrawerTask-v0` The agent must close the kitchen drawer in the scene.
 - Rearrangement open drawer: `RearrangeOpenDrawerTask-v0` The agent must open the kitchen drawer in the scene.
 - Rearrangement close fridge : `RearrangeCloseFridgeTask-v0` The agent must close the kitchen fridge in the scene.
 - Rearrangement open fridge : `RearrangeOpenFridgeTask-v0` The agent must open the kitchen fridge in the scene.
 - Rearrangement navigate to object : `NavToObjTask-v0` The agent must navigate to an object located at a known geometric position in the scene.
 - Rearrangement pick : `RearrangePickTask-v0` The agent must pick up a specific object in the scene from given the object's geometric coordinates.
 - Rearrangement place : `RearrangePlaceTask-v0` The agent must place a grasped object at  a geometric set of coordinates.
 - Rearrangement do nothing : `RearrangeEmptyTask-v0` The agent does not have to do anything. Useful for debugging.
 - Rearrangement reach : `RearrangeReachTask-v0` The agent must place its end effector into a specific location defined by geometric coordinates.
 - Rearrangement composite tasks : `RearrangePddlTask-v0` The agent must perform a sequence of sub-tasks in succession defined by a PDDL plan.


| Key | Description |
| --- | --- |
|habitat.task.type | The registered task that will be used. For example : `InstanceImageNav-v1` or `ObjectNav-v1`.
|habitat.task.physics_target_sps |  The size of each simulator physics update will be 1 / physics_target_sps. |
|habitat.task.reward_measure | The name of the Measurement that will correspond to the reward of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration (under `habitat.task.measurements`, see below for a list of available measurements). For example, `distance_to_goal_reward` for navigation or `place_reward` for the rearrangement place task.|
|habitat.task.success_measure | The name of the Measurement that will correspond to the success criteria of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration (under `habitat.task.measurements`, see below for a list of available measurements). If the measurement has a non-zero value, the episode is considered a success. |
|habitat.task.end_on_success | If True, the episode will end when the success measure indicates success. Otherwise the episode will go on (this is useful when doing hierarchical learning and the robot has to explicitly decide when to change policies)|
|habitat.task.task_spec |  When doing the `RearrangePddlTask-v0` only, will look for a pddl plan of that name to determine the sequence of sub-tasks that need to be completed. The format of the pddl plans files is undocumented.|
|habitat.task.task_spec_base_path |  When doing the `RearrangePddlTask-v0` only, the relative path where the task_spec file will be searched.|
|habitat.task.spawn_max_dists_to_obj| For `RearrangePickTask-v0` task only. Controls the maximum distance the robot can be spawned from the target object. |
| habitat.task.base_angle_noise| For Rearrangement tasks only. Controls the standard deviation of the random normal noise applied to the base's rotation angle at the start of an episode.|
| habitat.task.base_noise| For Rearrangement tasks only. Controls the standard deviation of the random normal noise applied to the base's position at the start of an episode.|

## Visual Agents
We define the visual agents as a bundle of sim sensors. They are a quick way to specify the visual observation space you need in your environment. You can inherit a default agent in your configuration.
Some sample agents are defined [here](./habitat/simulator/agents/) and are reused in many of our benchmarks. To use one of these agents, add
```
  - /habitat/simulator/agents@habitat.simulator.agents.main_agent: <the key of the agent>
```
to your defaults list. All agents have a camera resolution of 256 by 256.
| Key | Description |
| --- | --- |
|depth_head_agent| An Agent with a single depth camera attached to its head.|
|rgb_head_agent| An Agent with a single RGB camera attached to its head.|
|rgbd_head_rgbd_arm_agent| An Agent with both a RGB and depth camera attached to its head and arm.|



## Environment
Some habitat environment configurations.
| Key | Description |
| --- | --- |
|habitat.environment.max_episode_steps| The maximum number of environment steps before the episode ends.|
|habitat.environment.max_episode_seconds| The maximum number of wall-clock seconds before the episode ends.|

## Discrete Navigation Actions
Actions are the means through an agent affects the environment. They are defined in the dictionary `habitat.task.actions`. All actions in this dictionary will be available for the agent to perform. Note that Navigation action space is discrete while the Rearrangement action space is continuous. The way one would add an action to a configuration file would be by adding to the `defaults` list. For example:
```
defaults:
  - /habitat/task/actions:
    - move_forward
    - turn_left
```

| Key | Description |
| --- | --- |
| habitat.task.actions.stop |     In Navigation tasks only, the stop action is a discrete action. When called, the Agent will request to stop the navigation task. Note that this action is needed to succeed in a Navigation task since the Success is determined by the Agent calling the stop action within range of the target. Note that this is different from the RearrangeStopActionConfig that works for Rearrangement tasks only instead of the Navigation tasks.|
| habitat.task.actions.empty | In Navigation tasks only, the pass action. The robot will do nothing.|
| habitat.task.actions.move_forward |     In Navigation tasks only, this discrete action will move the robot forward by a fixed amount determined by the `habitat.simulator.forward_step_size` amount. |
| habitat.task.actions.turn_left |     In Navigation tasks only, this discrete action will rotate the robot to the left  by a fixed amount determined by the `habitat.simulator.turn_angle` amount. |
| habitat.task.actions.turn_right |     In Navigation tasks only, this discrete action will rotate the robot to the right by a fixed amount determined by the `habitat.simulator.turn_angle` amount. |
| habitat.task.actions.look_up |      In Navigation tasks only, this discrete action will rotate the robot's camera up by a fixed amount determined by the `tilt_angle` amount of the look_up action. |
| habitat.task.actions.look_down |      In Navigation tasks only, this discrete action will rotate the robot's camera down by a fixed amount determined by the `tilt_angle` amount  of the look_down action. |

## Navigation Measures
A measure is a way to collect data about the environment at each step that is not sensor information. Measures can contain privileged information for the user (like a top down map) or for training (like rewards).
The way one would add a measure to a configuration file would be by adding to the `defaults` list. For example:
```
defaults:
  - /habitat/task/measurements:
    - articulated_agent_force
    - force_terminate
```

| Key | Description |
| --- | --- |
| habitat.task.measurements.num_steps| In both Navigation and Rearrangement tasks, counts the number of steps since  the start of the episode.|
| habitat.task.measurements.distance_to_goal | In Navigation tasks only, measures the geodesic distance to the goal.|
|habitat.task.measurements.distance_to_goal.distance_to | If 'POINT' measures the distance to the closest episode goal. If 'VIEW_POINTS' measures the distance to the episode's goal viewpoints (useful in image nav). |
|habitat.task.measurements.success |     For Navigation tasks only, Measures 1.0 if the robot reached a success and 0 otherwise.  A success is defined as calling the `habitat.task.actions.stop` when the `habitat.task.measurements.distance_to_goal` Measure is smaller than `success_distance`.|
|habitat.task.measurements.success.success_distance| The minimal distance the robot must be to the goal for a success.|
|habitat.task.measurements.spl|    For Navigation tasks only, Measures the SPL (Success weighted by Path Length) ref: [On Evaluation of Embodied Agents - Anderson et. al](https://arxiv.org/pdf/1807.06757.pdf).  Measure is always 0 except at success where it will be  the ratio of the optimal distance from start to goal over the total distance  traveled by the agent. Maximum value is 1. `SPL = success * optimal_distance_to_goal / distance_traveled_so_far`
|habitat.task.measurements.soft_spl |     For Navigation tasks only, Similar to SPL, but instead of a boolean, success is now calculated as 1 - (ratio of distance covered to target).   `SoftSPL = max(0, 1 - distance_to_goal / optimal_distance_to_goal) * optimal_distance_to_goal / distance_traveled_so_far`
|habitat.task.measurements.distance_to_goal_reward    |    In Navigation tasks only, measures a reward based on the distance towards the goal. The reward is `- (new_distance - previous_distance)` i.e. the decrease of distance to the goal.

## Navigation Lab Sensors
Lab sensors are any non-rendered sensor observation, like geometric goal information. The way one would add a sensor to a configuration file would be by adding to the `defaults` list. For example:
```
defaults:
  - /habitat/task/lab_sensors:
    - objectgoal_sensor
    - compass_sensor
```

| Key | Description |
| --- | --- |
|  habitat.task.lab_sensors.objectgoal_sensor|  For Object Navigation tasks only. Generates a discrete observation containing the id of the goal object for the episode. |
|  habitat.task.lab_sensors.objectgoal_sensor.goal_spec| A string that can take the value TASK_CATEGORY_ID or OBJECT_ID. If the value is TASK_CATEGORY_ID, then the observation will be the id of the `episode.object_category` attribute, if the value is OBJECT_ID, then the observation will be the id of the first goal object. |
|  habitat.task.lab_sensors.objectgoal_sensor.goal_spec_max_val| If the ` habitat.task.lab_sensors.objectgoal_sensor.goal_spec` is OBJECT_ID, then `goal_spec_max_val` is the total number of different objects that can be goals. Note that this value must be greater than the largest episode goal category id. |
|  habitat.task.lab_sensors.instance_imagegoal_sensor  |    Used only by the InstanceImageGoal Navigation task. The observation is a rendered image of the goal object within the scene.|
|  habitat.task.lab_sensors. instance_imagegoal_hfov_sensor |     Used only by the InstanceImageGoal Navigation task. The observation is a single float value corresponding to the Horizontal field of view (HFOV) in degrees of  the image provided by the `habitat.task.lab_sensors.instance_imagegoal_sensor `.|
|  habitat.task.lab_sensors.compass_sensor |     For Navigation tasks only. The observation of the `EpisodicCompassSensor` is a single float value corresponding to the angle difference in radians between the current rotation of the robot and the start rotation of the robot along the vertical axis. |
|  habitat.task.lab_sensors.gps_sensor |     For Navigation tasks only. The observation of the EpisodicGPSSensor are two float values corresponding to the vector difference in the horizontal plane between the current position and the start position of the robot (in meters). |
| habitat.task.lab_sensors.pointgoal_with_gps_compass_sensor | Indicates the position of the point goal in the frame of reference of the robot. |


## Continuous Rearrangement Actions

| Key | Description |
| --- | --- |
|habitat.task.actions.arm_action |In Rearrangement tasks only, the action that will move the robot arm around. The action represents to delta angle (in radians) of each joint. |
|habitat.task.actions.arm_action.grasp_thresh_dist| The grasp action will only work on the closest object if its distance to the end effector is smaller than this value. Only for `MagicGraspAction` grip_controller.|
|habitat.task.actions.arm_action.grip_controller| Can either be None,  `MagicGraspAction` or `SuctionGraspAction`. If None, the arm will be unable to grip object. Magic grasp will grasp the object if the end effector is within grasp_thresh_dist of an object, with `SuctionGraspAction`, the object needs to be in contact with the end effector. |
|habitat.task.actions.base_velocity |     In Rearrangement only. Corresponds to the base velocity. Contains two continuous actions, the first one controls forward and backward motion, the second the rotation.
|habitat.task.actions.rearrange_stop     | In rearrangement tasks only, if the robot calls this action, the task will end.|
|habitat.task.actions.oracle_nav_action| Rearrangement Only, Oracle navigation action. This action takes as input a discrete ID which refers to an object in the PDDL domain. The oracle navigation controller then computes the actions to navigate to that desired object.|


## Rearrangement Sensors


| Key | Description |
| --- | --- |
|habitat.task.lab_sensors.relative_resting_pos_sensor     | Rearrangement only. Sensor for the desired relative position of the end-effector's resting position, relative to the end-effector's current position. The three values correspond to the cartesian coordinates of the desired resting position in the frame of reference of the end effector. The desired resting position is determined by the habitat.task.desired_resting_position coordinates relative to the robot's base.|
| habitat.task.lab_sensors.is_holding_sensor | Rearrangement only. A single float sensor with value 1.0 if the robot is grasping any object and 0.0 otherwise.|
|habitat.task.lab_sensors.end_effector_sensor | Rearrangement only. the cartesian coordinates (3 floats) of the arm's end effector in the frame of reference of the robot's base.|
|habitat.task.lab_sensors.joint_sensor| Rearrangement only. Returns the joint positions of the robot.|
| habitat.task.lab_sensors.goal_sensor |     Rearrangement only. Returns the relative position from end effector to a goal position in which the agent needs to place an object. |
| habitat.task.lab_sensors.target_start_gps_compass_sensor |     Rearrangement only. Returns the initial position of every object that needs to be rearranged in composite tasks relative to the robot's start position, in 2D polar coordinates (distance and angle in the horizontal plane). |
| habitat.task.lab_sensors.target_goal_gps_compass_sensor |    Rearrangement only. Returns the desired goal position of every object that needs to be rearranged in composite tasks relative to the robot's start position, in 2D polar coordinates (distance and angle in the horizontal plane). |

## Rearrangement Measures
| Key | Description |
| --- | --- |
|habitat.task.measurements.end_effector_to_rest_distance | Rearrangement only. Distance between current end effector position  and the resting position of the end effector. Requires that the RelativeRestingPositionSensor is attached to the agent (see Rearrangement Sensors above to see how to attach sensors). |
|habitat.task.measurements.articulated_agent_force |      The amount of force in newton's applied by the robot. It computes both the instantaneous and accumulated force during the episode. |
|habitat.task.measurements.does_want_terminate | Rearrangement Only. Measures 1 if the agent has called the stop action and 0 otherwise.    |
|habitat.task.measurements.force_terminate |    If the force is greater than a certain threshold, this measure will be 1.0 and 0.0 otherwise.   Note that if the measure is 1.0, the task will end as a result. |
|habitat.task.measurements.force_terminate.max_accum_force |  The threshold for the accumulated force before calling termination. -1 is no threshold, i.e., force-based termination is never called.   |
|habitat.task.measurements.force_terminate.max_instant_force |  The threshold for the current, instantaneous force before calling termination. -1 is no threshold, i.e., force-based termination is never called.   |
|habitat.task.measurements.object_to_goal_distance |  In rearrangement only. The distance between the target object and the goal position for the object. |
|habitat.task.measurements.obj_at_goal |  The measure is a dictionary of target indexes to float. The values are 1 if the object is within succ_thresh of the goal position for that object.   |
|habitat.task.measurements.obj_at_goal.succ_thresh | The threshold distance below which an object is considered at the goal location.    |
|habitat.task.measurements.art_obj_at_desired_state |   Rearrangement open/close container tasks only. Whether the articulated object (fridge or cabinet door) is in a desired joint state (open or closed) as defined by the task. |
|habitat.task.measurements.rot_dist_to_goal |  Rearrangement Navigation task only. The angle between the forward direction of the agent and the direction to the goal location.   |
|habitat.task.measurements.composite_stage_goals |       Composite Rearrangement only. 1.0 if the agent complete a particular stage defined in `stage_goals` and 0.0 otherwise. Stage goals are specified in the `pddl` task description.  |
|habitat.task.measurements.nav_to_pos_succ |     Rearrangement Navigation task only. The value is 1.0 if the robot is within success_distance of the goal position. |

### Task defining rearrangement measures

You can change the success and reward measures of a task by changing the `habitat.task.success_measure`  and the `habitat.task.reward_measure` keys respectively.

For Rearrangement Pick : `pick_success` | `place_reward`
For Rearrangement Place : `place_success` | `place_reward`
For Rearrangement Open / Close Articulated Object : `art_obj_success` | `art_obj_reward`
For Rearrangement Navigation : `nav_to_obj_success` | `nav_to_obj_reward`
For Composite Rearrangement using a PDDL plan : `pddl_compositesuccess` | ` move_objects_reward`
