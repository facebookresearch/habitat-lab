import gzip
import json
import glob
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fixing episode dataset as advised in https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md

data_dir = "home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/objectgoal_floorplanner_receptacles/val/content"
scene_dataset = "home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner/hab-fp.scene_dataset_config.json"

# data_dir = "home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/objectgoal_hm3d/val/content"
# scene_dataset = "home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"


for path in glob.glob(f"{data_dir}/*.gz"):
    print(path)

    with gzip.open(path, "rt") as f:
        scene_data = json.load(f)

    for episode in scene_data["episodes"]:
        episode["scene_dataset_config"] = scene_dataset

    with gzip.open(path, "w") as f:
        f.write(json.dumps(scene_data).encode())
