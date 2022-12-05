"""
This script is intended to run from the "src" root:
python home_robot/experimental/theo/habitat_projects/tasks/object_navigation/random_scripts/extract_image_instances.py
"""
import json
import gzip
import glob


if __name__ == "__main__":
    episode_dir = "home_robot/experimental/theo/habitat_projects/datasets/episode_datasets/imageinstancegoal_hm3d/val/content/*"
    for scene_path in glob.glob(episode_dir):
        with gzip.open(scene_path) as f:
            scene_data = json.loads(f.read())
            print(scene_path.keys())
