#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl
import glob
import numpy as np

def get_results_from_folder(folder_name):
    # Given a folder returns a dictionary with episode_id
    all_files = glob.glob(f"{folder_name}/*.pkl")
    print(len(all_files))
    results_dict = {}
    for file_name in all_files:
        with open(file_name, "rb") as f:
            curr_result = pkl.load(f)
        results_dict[curr_result["id"]] = EpisodeResults(curr_result["summary"])     
    return results_dict

class EpisodeResults():
    def __init__(self, episode_dict):
        self.episode_dict = episode_dict
        self.success = episode_dict["composite_success"]
        self.num_steps = episode_dict["num_steps"]
    
    def compute_time(self):
        return self.num_steps

class EvalRunResults():
    def __init__(self, folder_name):
        self.results_run = get_results_from_folder(folder_name)
        self.average_metrics()

    def compute_time(self, episode_id):
        return self.results_run[episode_id].compute_time()

    def average_metrics(self, metric_names=["composite_success", "num_steps"]):
        metric_list = {metric_name: [] for metric_name in metric_names}
        for episode_id, episode_result in self.results_run.items():
            
            for metric_name in metric_names:
                metric_list[metric_name].append(episode_result.episode_dict[metric_name])
        
        for metric_name in metric_names:
            mean = np.mean(metric_list[metric_name])
            std = np.std(metric_list[metric_name])
            print("{}: {:.02f} {:.02f}".format(metric_name, mean, std))

    def compute_speedup(self, cooperative_results):
        # Computes speed up with respect to base_eval_results
        # cooperative_results: List of EvalRunResults for different 
        # 1. We will only check the episodes where the single agent is successful
        speedup_all_seeds = []
        for cooperative_result in cooperative_results:
            speedups = []
            for episode_id in base_eval_results.results_run:
                time_single_agent =self.compute_time(episode_id)
                time_multi_agent = cooperative_result.compute_time(episode_id)
                assert base_eval_results.results_run[episode_id].success
                succes_multi = self.results_run[episode_id].success 
                curr_speedup = success_multi * time_single_agent / time_multi_agent
                speedups.append(curr_speedup)
        
            mean_speedup, std = np.mean(speedups), np.std(speedups)
            speedup_all_seeds.append(mean_speedup)
        mean, std = np.mean(speedup_all_seeds), np.std(speedup_all_seeds)
        print("Speedup: {:.02f} ({:.02f})".format(mean, std))

def compute_all_metrics():
    for path in paths:
        results = EvalRunResults(path)
        results.average_metrics()
        
    fpth = "multirun/2023-06-15/17-37-03/0/eval_stats"
    single_agent = EvalRunResults(fpth_single)
    for experiment in baselines:
        seed_exps = [EvalRunResults(exp_path_seed) for exp_path_seed in experiment]
        single_agent.compute_speedup(seed_exps)
    common_agent = EvalRunResults(fpth)
    common_agent.compute_speedup(single_agent)

if __name__ == '__main__':
    baselines = [
        "multirun/2023-06-18/02-12-56/0/eval_data/",
        "multirun/2023-06-18/02-13-01/0/eval_data/",
        "multirun/2023-06-19/16-02-14/0/eval_data/"
    ]
    for baseline in baselines:
        EvalRunResults(baseline)
