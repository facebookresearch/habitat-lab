# Configuration Keys

## Dataset
Configuration for the dataset of the task.
A dataset consists of episodes
(a start configuration for a task within a scene) and a scene dataset
(with all the assets needed to instantiate the task)

| Key | Description |
| --- | --- |
| habitat.dataset.type |  The key for the dataset class that will be used. Examples of such keys are `PointNav-v1`, `ObjectNav-v1`, `InstanceImageNav-v1` or `RearrangeDataset-v0`. Different datasets have different properties so you should use the dataset that fits your task. |
| habitat.dataset.scene_dir | The path to the directory containing the scenes that will be used. You should put all your scenes in the same folder (example `data/scene_datasets`) to avoid having to change it. |
|habtat.dataset.data_path | The path to the episode dataset. Episodes need to be compatible with the `type` argument (so they will load properly) and only use scenes that are present in the `scenes_dir`.|
|habitat.dataset.split | `data_path` can have a `split` in the path. For example: "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz" the value in "{split}" will be replaced by the value of the `split` argument. This allows to easily swap between training, validation and test episodes by only changing the split argument. |

## Task
The definition of the task in Habitat.

| Key | Description |
| --- | --- |
|habitat.task.type | The registered task that will be used. For example : `InstanceImageNav-v1` or `ObjectNav-v1`
|habitat.task.reward_measure | The name of the Measurement that will correspond to the reward of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration. For example, `distance_to_goal_reward` for navigation or `place_reward` for the rearrangement place task. |
|habitat.task.success_measure | The name of the Measurement that will correspond to the success criteria of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration. If the measurement has a non-zero value, the episode is considered a success. |
|habitat.task. end_on_success | If True, the episode will end when the success measure indicates success. Otherwise the episode will go on (this is useful when doing hierarchical learning and the robot has to explicitly decide when to change policies)|

## Environment
Some habitat environment configurations.
| Key | Description |
| --- | --- |
|habitat.environment.max_episode_steps| The maximum number of environment steps before the episode ends.|
|habitat.environment.max_episode_seconds| The maximum number of seconds steps before the episode ends.|

## Navigation Actions
The way one would add an action to a configuration file would be by adding to the `defaults` list. For example:
```
defaults:
  - /habitat/task/actions:
    - move_forward
    - turn_left
```

| Key | Description |
| --- | --- |
| habitat.task.actions.stop |     In Navigation tasks only, the stop action is a discrete action. When called, the Agent will request to stop the navigation task. Note that this action is needed to succeed in a Navigation task since the Success is determined by the Agent calling the stop action within range of the target. Note that this is different from the RearrangeStopActionConfig that works for Rearrangement tasks only instead of the Navigation tasks.|
| habitat.task.actions.move_forward |     In Navigation tasks only, this discrete action will move the robot forward by a fixed amount determined by the `habitat.simulator.forward_step_size` amount. |
| habitat.task.actions.turn_left |     In Navigation tasks only, this discrete action will rotate the robot to the left  by a fixed amount determined by the `habitat.simulator.turn_angle` amount. |
| habitat.task.actions.turn_right |     In Navigation tasks only, this discrete action will rotate the robot to the right by a fixed amount determined by the `habitat.simulator.turn_angle` amount. |
| habitat.task.actions.look_up |      In Navigation tasks only, this discrete action will rotate the robot's camera up by a fixed amount determined by the `habitat.simulator.tilt_angle` amount. |
| habitat.task.actions.look_down |      In Navigation tasks only, this discrete action will rotate the robot's camera down by a fixed amount determined by the `habitat.simulator.tilt_angle` amount. |

## Navigation Measures
The way one would add a measure to a configuration file would be by adding to the `defaults` list. For example:
```
defaults:
  - /habitat/task/measurements:
    - robot_force
    - force_terminate
```

| Key | Description |
| --- | --- |
| habitat.task.measurements.num_steps| In both Navigation and Rearrangement tasks, counts the number of steps since  the start of the episode.|
| habitat.task.measurements.distance_to_goal | In Navigation tasks only, measures the geodesic distance to the goal.|
|habitat.task.measurements.distance_to_goal.distance_to | If 'POINT' measures the distance to the closest episode goal. If 'VIEW_POINTS' measures the distance to the episode's goal's viewpoint. |
|habitat.task.measurements.success |     For Navigation tasks only, Measures 1.0 if the robot reached a success and 0 otherwise.  A success is defined as calling the `habitat.task.actions.stop` when the `habitat.task.measurements.distance_to_goal` Measure is smaller than `success_distance`.|
|habitat.task.measurements.success.success_distance| The minimal distance the robot must be to the goal for a success.|
|habitat.task.measurements.spl|    For Navigation tasks only, Measures the SPL (Success weighted by Path Length) ref: [On Evaluation of Embodied Agents - Anderson et. al](https://arxiv.org/pdf/1807.06757.pdf).  Measure is always 0 except at success where it will be  the ratio of the optimal distance from start to goal over the total distance  traveled by the agent. Maximum value is 1. `SPL = success * optimal_distance_to_goal / distance_traveled_so_far`
|habitat.task.measurements.soft_spl |     For Navigation tasks only, Similar to SPL, but instead of a boolean, success is now calculated as 1 - (ratio of distance covered to target).   `SoftSPL = max(0, 1 - distance_to_goal / optimal_distance_to_goal) * optimal_distance_to_goal / distance_traveled_so_far`
|habitat.task.measurements.distance_to_goal_reward    |    In Navigation tasks only, measures a reward based on the distance towards the goal. The reward is `- (new_distance - previous_distance)` i.e. the decrease of distance to the goal.

## Navigation Lab Sensors
Lab sensors are any non-rendered sensor observation. Like geometric goal information. The way one would add a sensor to a configuration file would be by adding to the `defaults` list. For example:
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
