#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import pickle as pkl
from typing import Any, Dict

import numpy as np

METRICS_INTEREST = ["composite_success", "num_steps", "num_steps_fail"]
MAX_NUM_STEPS = 1000


# TODO: this should go into utilities
def pretty_print(metric_dict, latex=False, metric_names=None):
    def get_number_str(mean, std):
        if latex:
            return "${:0.2f}_{{ \\pm {:0.2f} }}$".format(mean, std)
        else:
            return "{:0.2f} \u00B1 {:0.2f}  ".format(mean, std)

    result_str = []
    if metric_names is None:
        metric_names = list(metric_dict)
    for metric_name in metric_names:
        mean, std = metric_dict[metric_name]
        number_str = get_number_str(mean, std)
        if not latex:
            curr_metric_str = f"{metric_name}: {number_str}"
        else:
            curr_metric_str = number_str

        result_str.append(curr_metric_str)

    if latex:
        return " & ".join(result_str)
    else:
        return " ".join(result_str)


def get_episode_info(file_name):
    # Read a single pickle file with results from an episode/seed
    with open(file_name, "rb") as f:
        curr_result = pkl.load(f)

    metrics = {
        metric_name: curr_result["summary"][metric_name]
        for metric_name in METRICS_INTEREST
        if metric_name in curr_result["summary"]
    }
    if "num_steps_fail" in METRICS_INTEREST:
        if metrics["composite_success"] == 1:
            num_steps_fail = metrics["num_steps"]
        else:
            num_steps_fail = MAX_NUM_STEPS
        metrics["num_steps_fail"] = num_steps_fail

    return int(curr_result["id"]), metrics


def aggregate_per_episode_dict(dict_data, average=False, std=False):
    # Given a dictionary where every episode has a list of metrics
    # Returns a dict with the metrics aggregated per episode
    new_dict_data = {}
    for episode_id, episode_data in dict_data.items():
        metric_names = list(episode_data[0].keys())
        results_aggregation = {}
        for metric_name in metric_names:
            metric_list = np.array(
                [sample_data[metric_name] for sample_data in episode_data]
            )
            if not average:
                result = metric_list
            else:
                if std:
                    result = [np.mean(metric_list), np.std(metric_list)]
                else:
                    result = np.mean(metric_list)

            results_aggregation[metric_name] = result
        new_dict_data[episode_id] = results_aggregation
    return new_dict_data


def get_checkpoint_results(ckpt_path):
    # Reads files from folder ckpt_path
    # and averages different runs of the same episode
    all_files = glob.glob(f"{ckpt_path}/*")
    dict_results: Dict[str, Any] = {}
    for file_name in all_files:
        episode_id, episode_info = get_episode_info(file_name)
        if episode_id not in dict_results:
            dict_results[episode_id] = []

        dict_results[episode_id].append(episode_info)

    # Potentially verify here that no data is missing
    dict_results = aggregate_per_episode_dict(dict_results, average=True)
    return dict_results


def relative_metric(episode_baseline_data, episode_solo_data):
    assert episode_solo_data["composite_success"] == 1

    composite_success = episode_baseline_data["composite_success"]
    efficiency = (
        episode_solo_data["num_steps"] / episode_baseline_data["num_steps"]
    )
    RE = composite_success * efficiency * 100
    REMT = (
        100.0
        * episode_solo_data["num_steps"]
        / episode_baseline_data["num_steps_fail"]
    )
    composite_success *= 100
    return {"composite_success": composite_success, "RE": RE, "RE_MT": REMT}


def compute_relative_metrics(per_episode_baseline_dict, per_episode_solo_dict):
    # Computes the aggregated metrics coming from a particular training checkpoint.
    res_dict = {}
    all_results = []
    for episode_id in per_episode_solo_dict:
        episode_solo_data = per_episode_solo_dict[episode_id]
        episode_baseline_data = per_episode_baseline_dict[episode_id]
        curr_metric = relative_metric(episode_baseline_data, episode_solo_data)
        all_results.append(curr_metric)
        res_dict[episode_id] = curr_metric

    average_over_episodes = aggregate_per_episode_dict(
        {"all_episodes": all_results}, average=True
    )
    return average_over_episodes["all_episodes"]


def compute_relative_metrics_multi_ckpt(
    experiments_path_dict, solo_path, latex=False
):
    # Computes and prints metrics for all baselines
    # given the path of the solo episodes, and a dictionary baseline_name: path_res_baselines
    all_results = []
    solo = get_checkpoint_results(solo_path)
    for baseline_name, baselines_path in experiments_path_dict.items():
        for baseline_path in baselines_path:
            baseline = get_checkpoint_results(baseline_path)
            curr_res = compute_relative_metrics(baseline, solo)
            all_results.append(curr_res)

        results_baseline = aggregate_per_episode_dict(
            {"all_episodes": all_results}, average=True, std=True
        )["all_episodes"]

        metrics_str = pretty_print(results_baseline, latex=latex)
        print(f"{baseline_name}: {metrics_str}")


def compute_all_metrics(latex_print=False):
    root_dir = "/fsx-siro/akshararai/hab3/zsc_eval/20_ep_data"
    solo_path = f"{root_dir}/solo_eval_data"
    experiments_path = {
        "GT_coord": [
            f"{root_dir}/GTCoord_eval_data",
            f"{root_dir}/GTCoord_eval_data",
            f"{root_dir}/GTCoord_eval_data",
        ],
        "Pop_play": [
            f"{root_dir}/GTCoord_eval_data",
            f"{root_dir}/GTCoord_eval_data",
            f"{root_dir}/GTCoord_eval_data",
        ],
    }

    compute_relative_metrics_multi_ckpt(
        experiments_path, solo_path, latex=latex_print
    )


if __name__ == "__main__":
    print("\n\nResults")

    compute_all_metrics(latex_print=False)
    print("\n\nLATEX")
    compute_all_metrics(latex_print=True)
