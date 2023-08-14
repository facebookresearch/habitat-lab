# Generating episodes for Open Vocab Mobile Manipulation task

To generate episodes for the OVMM task, we need to run the `habitat-lab/habitat/datasets/rearrange/run_episode_generator.py` followed by the `habitat-lab/habitat/datasets/rearrange/modify_episodes_for_object_rearrange.py` scripts.

The `habitat-lab/habitat/datasets/rearrange/generate_ovmm_episode_inits.py` script adds randomly sampled agent start poses to each episode for deterministic evaluations.

After this, there are several utility scripts in [this repository](https://github.com/JHurricane96/lang_rearrange_scripts) that can be run:
- `merge_episodes.py` merges episode files together. This is useful when episodes are generated in parallel, say by scene.
- `decompose_episode_matrices.py` takes viewpoints and object transformation matrices and stores them separately in dense matrix `.npy` files, and leaves behind only pointers in the episode file. This greatly reduces memory requirements when loading the episode dataset since these matrices can be loaded lazily per episode.
- `split_dataset.py` splits a single episode file by scene into multiple smaller episode files. This is useful when using large episode datasets during training.

The following sections describe some of these scripts in more detail:

## OVMM episode modification script

Given the rearrange episodes data at <data_dir>/<split>/<geogoal_episodes_name>.yaml, the script `habitat-lab/habitat/datasets/rearrange/modify_episodes_for_object_rearrange.py` can be used to generate the corresponding episodes for Open Vocab Mobile Manipulation task. The command for using this script is as follows:

```
python habitat-lab/habitat/datasets/rearrange/modify_episodes_for_object_rearrange.py \
--source_data_dir /geogoal/episodes/directory \
--target_data_dir /target/episodes/directory \
--source_episodes_tag <geogoal_episodes_name> \
--target_episodes_tag <target_episodes_name> \
--obj_category_mapping_file /path/to/csv/with/object_category_mapping \
--rec_category_mapping_file /path/to/csv/with/receptacle_category_mapping \
--rec_cache_dir /path/to/receptacle/cache \
--num_episodes <number of episodes to modify> \
--config /path/to/config \
--add_viewpoints
```

The csv files should contain a field `clean_category` that specifies category of the object model with name `name`. The default receptacle cache directory is `data/cache/receptacle_viewpoints`. The config file is the same as is used in the geo goal episode generation step.

## Saving episode start poses
```
python habitat-lab/habitat/datasets/rearrange/generate_ovmm_episode_inits.py habitat.dataset.data_path=/path/to/episodes
```
The new episodes with start poses will be saved in the same directory with `-with_init_poses` suffix.