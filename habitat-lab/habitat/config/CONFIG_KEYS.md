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
|habitat.dataset.split | `data_path` can have a `split` in the path. For example: "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz" the value in "{split}" will be replaced by the value of the `split` argument. This allows to easily swap between `train` and `eval` episodes by only changing the split argument.|

# Task
The definition of the task in Habitat.

| Key | Description |
| --- | --- |
|habitat.task.type | The registered task that will be used. For example : `InstanceImageNav-v1` or `ObjectNav-v1`
|habitat.task.reward_measure | The name of the Measurement that will correspond to the reward of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration. |
|habitat.task.success_measure | The name of the Measurement that will correspond to the success criteria of the robot. This value must be a key present in the dictionary of Measurements in the habitat configuration. If the measurement has a non-zero value, the episode is considered a success. |
|habitat.task. end_on_success | If True, the episode will end when the success measure indicates success. Otherwise the episode will go on (this is useful when doing hierarchical learning and the robot has to explicitly decide when to change policies)|
