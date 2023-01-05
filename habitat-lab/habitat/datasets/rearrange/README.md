## Generating episodes for Object Rearrangement task

Given the standard GeoGoal rearrange episodes data at <data_dir>/<split>/<geogoal_episodes_name>.yaml, the script `convert_to_object_rearrange_episodes.py` can be used to generate the corresponding episodes for Object Rearrangement task. The command for using this script is as follows:

```
python habitat-lab/habitat/datasets/rearrange/convert_to_object_rearrange_episodes.py --data_dir <data_dir> --source_episodes_tag <geogoal_episodes_name> --target_episodes_tag <target_episodes_name> --obj_category_mapping_file /path/to/csv/with/object_category_mapping --rec_category_mapping_file /path/to/csv/with/receptacle_category_mapping
```

The csv files should contain a field `clean_category` that specifies category of the object model with name `name`.

For example to generate Object Rearrangement version of the ReplicaCAD episodes, run the following:

```
python habitat-lab/habitat/datasets/rearrange/convert_to_object_rearrange_episodes.py --data_dir data/datasets/replica_cad/rearrange/v1/ --source_episodes_tag rearrange_easy --target_episodes_tag categorical_rearrange_easy
```
