"""
This script is intended to run from the "src" root:
python home_robot/experimental/theo/habitat_projects/tasks/object_navigation/random_scripts/aggregate_eval_metrics_per_scene.py
"""
import json
import numpy as np
from pprint import pprint
from collections import defaultdict


if __name__ == "__main__":
    results_paths = [
        "home_robot/experimental/theo/habitat_projects/tasks/object_navigation/datadump/results/eval_floorplanner/val_episode_results.json",
        "home_robot/experimental/theo/habitat_projects/tasks/object_navigation/datadump/results/eval_floorplanner/train_episode_results.json"
    ]
    scene_results = defaultdict(list)
    for path in results_paths:
        episode_results = json.load(open(path, "r"))
        for k, v in episode_results.items():
            scene_id = "_".join(k.split("_")[:-1])
            scene_results[scene_id].append(v["success"])
    scene_results = dict(scene_results)
    scene_results = {k: {"success": np.mean(v), "episodes": len(v)} for k, v in scene_results.items()}
    pprint(scene_results, indent=4)
