#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import pickle as pkl
from typing import Any, Dict

import numpy as np
import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import os
import json
from tqdm import tqdm


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
    json_name = "/fsx-siro/xavierpuig/eval_data_akshara/" + file_name + ".json"
    
    if os.path.isfile(json_name):
        with open(json_name, "r") as f:
            data = json.load(f)
        return data["id"], data["metrics"]


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

    base_dir = os.path.dirname(json_name)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(json_name, "w+") as f:
        dict_res = {
            "id": int(curr_result["id"]),
            "metrics": metrics
        }
        f.write(json.dumps(dict_res))   
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
                    result = (np.mean(metric_list), np.std(metric_list))
                else:
                    result = np.mean(metric_list)

            results_aggregation[metric_name] = result
        new_dict_data[episode_id] = results_aggregation
    return new_dict_data

def process_file(file_name):
    episode_id, episode_info = get_episode_info(file_name)
    return (episode_id, episode_info)

def get_dir_name(file_name):
    return os.path.dirname("/fsx-siro/xavierpuig/eval_data_akshara/"+file_name)

def get_checkpoint_results(ckpt_path):
    # Reads files from folder ckpt_path
    # and averages different runs of the same episode
    t1 = time.time()
    if type(ckpt_path) is str:
        all_files = glob.glob(f"{ckpt_path}/*")
    else:
        all_files = []
        for ck_path in ckpt_path:
            all_files += glob.glob(f"{ck_path}/*")
    print(len(all_files))
    dict_results: Dict[str, Any] = {}
    
    # Create folders:
    num_proc = 24
    pool = Pool(num_proc)
    res = pool.map(get_dir_name, all_files)
    pool.close()
    pool.join()
    res = list(set(res))
    
    for elem in res:
        if not os.path.exists(elem):
            os.makedirs(elem)
    
    num_proc = 24
    pool = Pool(num_proc)
    res = pool.map(process_file, all_files)
    pool.close()
    pool.join()

    # num_threads = 24  # You can adjust this value based on your system's capabilities
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     res = executor.map(process_file, all_files)

    for (episode_id, episode_info) in res:
        if episode_id not in dict_results:
            dict_results[episode_id] = []

        dict_results[episode_id].append(episode_info)

    # print(time.time() - t1)
    # Potentially verify here that no data is missing
    dict_results = aggregate_per_episode_dict(dict_results, average=True)
    # print(time.time() - t1)
    print(time.time() - t1, len(all_files))
    # print('__')
    return dict_results


def relative_metric(episode_baseline_data, episode_solo_data):
    try:
        assert episode_solo_data["composite_success"] == 1
    except:
        # print("Failed episode")
        return {}
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
        if episode_id not in per_episode_baseline_dict:
            # TODO: raise exception here
            continue
        episode_baseline_data = per_episode_baseline_dict[episode_id]
        curr_metric = relative_metric(episode_baseline_data, episode_solo_data)
        if len(curr_metric) == 0:
            # TODO: raise exception here
            continue
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
        print(f"Computing {baseline_name}...")
        for baseline_path in tqdm(baselines_path):
            if type(baselines_path) == list:
                baseline = get_checkpoint_results(baseline_path)
            
            elif type(baselines_path) == dict:
                
                baseline = get_checkpoint_results(baselines_path[baseline_path])
            else:
                raise Exception
            
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

def extend_exps_zsc(dict_exps):
    # Increases the experiments to include info of different agents
    new_experiments_path = {}
    for exp_name, paths in dict_exps.items():
        paths_exp = []
        dict_paths = {}
        for path in paths:
            files = glob.glob(f"{path}/eval_data*")
            dict_paths[path] = files
        new_experiments_path[exp_name] = dict_paths
    return new_experiments_path

def compute_all_metrics_zsc(latex_print=False):
    # root_dir = "/fsx-siro/akshararai/hab3/zsc_eval/20_ep_data"
    root_dir = "/fsx-siro/akshararai/hab3/zsc_eval/zsc_eval_data"
    
    
    solo_path = "/fsx-siro/akshararai/hab3/eval_solo/eval_data_multi_ep_speed_10"
    experiments_path = {
        "GT_coord": [
            f"{root_dir}/speed_5/GTCoord/2023-08-19/00-07-24/0",
            f"{root_dir}/speed_5/GTCoord/2023-08-19/00-07-24/1",
            f"{root_dir}/speed_5/GTCoord/2023-08-19/00-07-24/2",
        ],
        "Pop_play": [
            f"{root_dir}/speed_5/pp8/2023-08-19/00-05-08/0",
            f"{root_dir}/speed_5/pp8/2023-08-19/00-05-08/1",
            f"{root_dir}/speed_5/pp8/2023-08-19/00-05-08/2",
        ],
        "Plan_play_-1": [
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/3",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/7",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/11",
        ],

        "Plan_play_-2": [
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/2",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/6",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/10",
        ],

        "Plan_play_-3": [
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/1",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/5",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/9",
        ],
        "Plan_play_-4": [
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/0",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/4",
            f"{root_dir}/speed_5/plan_play/2023-08-25/18-19-41/8",
        ],
    }
    experiments_path = extend_exps_zsc(experiments_path)
    
    compute_relative_metrics_multi_ckpt(
        experiments_path, solo_path, latex=latex_print
    )


if __name__ == "__main__":
    print("\n\nResults")
    compute_all_metrics(latex_print=False)
    breakpoint()
    compute_all_metrics_zsc(latex_print=False)
    breakpoint()
    print("\n\nLATEX")
    compute_all_metrics(latex_print=True)
