## Generating episodes for Open Vocab Mobile Manipulation task

Given the rearrange episodes data at <data_dir>/<split>/<geogoal_episodes_name>.yaml, the script `habitat-lab/habitat/datasets/rearrange/modify_episodes_for_object_rearrange.py` can be used to generate the corresponding episodes for Open Vocab Mobile Manipulation task. The command for using this script is as follows:

```
python habitat-lab/habitat/datasets/rearrange/modify_episodes_for_object_rearrange.py --data_dir <data_dir> --source_episodes_tag <geogoal_episodes_name> --target_episodes_tag <target_episodes_name> --obj_category_mapping_file /path/to/csv/with/object_category_mapping --rec_category_mapping_file /path/to/csv/with/receptacle_category_mapping
```

The csv files should contain a field `clean_category` that specifies category of the object model with name `name`.

## Saving episode start poses
```
python habitat-lab/habitat/datasets/rearrange/generate_ovmm_episode_inits.py habitat.dataset.data_path=/path/to/episodes
```
The new episodes with start poses will be saved in the same directory with `-with_init_poses` suffix.