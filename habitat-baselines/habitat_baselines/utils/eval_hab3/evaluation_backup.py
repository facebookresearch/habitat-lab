#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl
import glob
import os
import numpy as np

def get_results_from_folder(folder_name):
    # Given a folder returns a dictionary with episode_id
    all_files = glob.glob(f"{folder_name}/*.pkl")
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
    def __init__(self, folder_name=None, eval_results=[], debug=False):
        self.debug = debug
        if folder_name is not None:
            self.results_run = get_results_from_folder(folder_name)
            self.folder_name = folder_name
        elif len(eval_results) > 0:
            self.results_run = self.aggregate(eval_results)
            self.folder_name = ""
        self.average_metrics()

    def compute_time(self, episode_id):
        return self.results_run[episode_id].compute_time()

    @property
    def average_success_rate(self):
        return self.average_metrics(metric_names=["composite_success"])["composite_success"]

    def average_metrics(self, metric_names=["composite_success", "num_steps"]):
        metric_list = {metric_name: [] for metric_name in metric_names}
        for episode_id, episode_result in self.results_run.items():
            
            for metric_name in metric_names:
                metric_list[metric_name].append(episode_result.episode_dict[metric_name])
        
        result_dict = {}
        for metric_name in metric_names:
            mean = np.mean(metric_list[metric_name])
            std = np.std(metric_list[metric_name])
            if self.debug:
                print("{}: {:.02f} {:.02f}".format(metric_name, mean, std))
            result_dict[metric_name] = [mean, std]
        return result_dict

    def compute_speedup(self, cooperative_result):
        # Computes speed up with respect to base_eval_results
        # cooperative_results: List of EvalRunResults for different 
        # 1. We will only check the episodes where the single agent is successful
        speedup_all_seeds = []
        base_eval_results = self

        speedups = []
        for episode_id in base_eval_results.results_run:
            if episode_id not in cooperative_result.results_run:
                continue
            time_single_agent = base_eval_results.compute_time(episode_id)
            time_multi_agent = cooperative_result.compute_time(episode_id)
            assert base_eval_results.results_run[episode_id].success
            success_multi = cooperative_result.results_run[episode_id].success 
            curr_speedup = success_multi * time_single_agent / time_multi_agent
            speedups.append(curr_speedup)
    
        mean, std = np.mean(speedups), np.std(speedups)
            
        if self.debug:
            print("Speedup: {:.02f} {:.02f}".format(mean, std))
        return [mean, std]

    def aggregate(self, list_eval_results):
        # Aggregates results across seeds and 
        episode_ids = list_eval_results[0].results_run.keys()
        results_run = {}
        for episode_id in episode_ids:
            metrics = list_eval_results[0].results_run[episode_id].episode_dict.keys()
            aggregated_dict = {}
            results_from_episode = []
            for eval_result in list_eval_results:
                if episode_id not in eval_result.results_run:
                    if self.debug:
                        print(f"Missing episode {episode_id} in file {eval_result.folder_name}")
                    continue
                results_from_episode.append(eval_result.results_run[episode_id])
            for key_metric in metrics:    
                metric_list = [result_episode.episode_dict[key_metric] for result_episode in results_from_episode]    
                aggregated_dict[key_metric] = np.mean(metric_list)
            results_run[episode_id] = EpisodeResults(aggregated_dict)
        return results_run


def print_metrics_aggregated(metrics_aggregated, print_latex=False, print_pretty=False):
    latex_res = []        
    for split in ["train", "val"]:
        if split not in metrics_aggregated:
            continue
        for metric_name in metrics_aggregated[split]:
            values = np.array(metrics_aggregated[split][metric_name]) * 100
            if print_pretty:
                print("{}.{} - mean: {:.02f} std: {:.02f}".format(split, metric_name, np.mean(values), np.std(values)))
                print("{}.{} - mean: {:.02f} std: {:.02f}".format(split, metric_name, np.mean(values), np.std(values)))
            # breakpoint()
            latex_res.append("${:0.2f}_{{ \\pm {:0.2f} }}$".format(np.mean(values), np.std(values)))
    
    latex_str = " & ".join(latex_res)
    if print_latex:
        print(latex_str)

def compute_all_metrics():
    print_latex = True
    print_pretty = True
    # Single agent runs:
    base_path = "/data/home/xavierpuig/habitat-lab/"
    single_agent = "multirun/2023-06-21/17-12-35/"

    ###############################
    # Get the single agent_run
    results_single_agent_seed = []
    for seed in range(3):
        for run_single_agent_file in glob.glob(f"{base_path}/{single_agent}/{seed}/eval_data"):
            results_single_agent_seed.append(EvalRunResults(run_single_agent_file))

    aggregated_results_single_agent = EvalRunResults(eval_results=results_single_agent_seed)
    ##############################


    #####
    # GT Coord
    pop_planner = "multirun/2023-06-22/00-34-44"
    for seed_number in range(3):
        run_id = seed_number
        path_run = f"{base_path}/{pop_planner}/{run_id}"
        print(path_run)
        if not os.path.isdir(path_run):
            continue
        
        results_train_pop = EvalRunResults(f"{path_run}/episode_data_train_pop")
        results_val_pop = EvalRunResults(f"{path_run}/episode_data_val2_pop")
        metrics_aggregated = {
            "train": {
                "SR": [],
                "SpeedUp": []
            },
            "val": {
                "SR": [],
                "SpeedUp": []
            }
        }
        metrics_aggregated["train"]["SpeedUp"].append(aggregated_results_single_agent.compute_speedup(results_train_pop)[0])
        metrics_aggregated["train"]["SR"].append(results_train_pop.average_success_rate[0])
        
        metrics_aggregated["val"]["SpeedUp"].append(aggregated_results_single_agent.compute_speedup(results_val_pop)[0])
        metrics_aggregated["val"]["SR"].append(results_val_pop.average_success_rate[0])
        
    print_metrics_aggregated(metrics_aggregated, print_latex=print_latex, print_pretty=print_pretty)           
    

    #####

    pop_planner = "multirun/2023-06-21/00-58-01"
    for plan_idx in range(4):
        # plan_idx = 3 - planner_partner
         # -(planner_partner + 1)
        planner_partner = 3 - plan_idx
        print("Planer Partner", plan_idx)
        metrics_aggregated = {
            "train": {
                "SR": [],
                "SpeedUp": []
            },
            "val": {
                "SR": [],
                "SpeedUp": []
            }
        }
        for seed_number in range(3):
            run_id = planner_partner * 3 + seed_number
            path_run = f"{base_path}/{pop_planner}/{run_id}"
            print(path_run)
            if not os.path.isdir(path_run):
                continue
            results_train_pop = EvalRunResults(f"{path_run}/episode_data_train_pop")
            results_val_pop = EvalRunResults(f"{path_run}/episode_data_val2_pop")

            metrics_aggregated["train"]["SpeedUp"].append(aggregated_results_single_agent.compute_speedup(results_train_pop)[0])
            metrics_aggregated["val"]["SpeedUp"].append(aggregated_results_single_agent.compute_speedup(results_val_pop)[0])
            metrics_aggregated["train"]["SR"].append(results_train_pop.average_success_rate[0])    
            metrics_aggregated["val"]["SR"].append(results_val_pop.average_success_rate[0])
        
            # Aggregate across seeds:
            #print(path_run)
        
        latex_res = []
        print_metrics_aggregated(metrics_aggregated, print_latex=print_latex, print_pretty=print_pretty)        
        # for split in metrics_aggregated:
        #     for metric_name in metrics_aggregated[split]:
        #         values = np.array(metrics_aggregated[split][metric_name]) * 100
        #         if print_pretty:
        #             print("{}.{} - mean: {:.02f} std: {:.02f}".format(split, metric_name, np.mean(values), np.std(values)))
        #             print("{}.{} - mean: {:.02f} std: {:.02f}".format(split, metric_name, np.mean(values), np.std(values)))
        #         # breakpoint()
        #         latex_res.append("${:0.2f}_{{ \\pm {:0.2f} }}$".format(np.mean(values), np.std(values)))
        
        # latex_str = " & ".join(latex_res)
        # if print_latex:
        #     print(latex_str)

    
    metrics_aggregated = {
        "val": {
            "SR": [],
            "SpeedUp": []
        }
    }
    # Planner-Planner
    pop_planner = "multirun/planner_robot_eval_2"
    print("Planner Planner")
    for seed_number in range(3):
        run_id = seed_number
        path_run = f"{base_path}/{pop_planner}/{run_id}"
        print(path_run)
        if not os.path.isdir(path_run):
            continue
        
        results_val_pop = EvalRunResults(f"{path_run}/episode_data_train_pop")

        metrics_aggregated["val"]["SpeedUp"].append(aggregated_results_single_agent.compute_speedup(results_val_pop)[0])
        metrics_aggregated["val"]["SR"].append(results_val_pop.average_success_rate[0])
        
    print_metrics_aggregated(metrics_aggregated, print_latex=print_latex, print_pretty=print_pretty)           
    

if __name__ == '__main__':
    compute_all_metrics()
    