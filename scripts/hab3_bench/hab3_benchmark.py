# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing
import os
import random
import time
from collections import defaultdict
from sys import platform

import numpy as np

import habitat
from habitat.core.spaces import ActionSpace


def create_env(args, proc_i):
    procs_per_gpu = args.n_procs // args.n_gpus
    procs_to_gpu = {i: i // procs_per_gpu for i in range(args.n_procs)}
    spec_gpu = procs_to_gpu[proc_i]
    print(f"assigning {proc_i} gpu {spec_gpu}")

    set_opts = args.opts
    set_opts.extend(
        [f"habitat.simulator.habitat_sim_v0.gpu_device_id={spec_gpu}"]
    )

    config = habitat.get_config(args.cfg, set_opts)
    return habitat.Env(config=config)


_barrier = None


class ActionSpaceWrapper(ActionSpace):
    def sample(self, count):
        # custom action sampler, this is because the default
        # sampler only samples one action at a time. Here we want to
        # sample actions from both agents
        all_actions = []
        all_action_args = {}
        for action_name, action_args in self.spaces.items():
            all_actions.append(action_name)
            for action_arg_name, action_arg in action_args.items():
                if "oracle_nav" in action_arg_name:
                    action_arg_value = (
                        np.array([random.randint(0, self.num_items_nav)]) + 1
                    )

                elif "humanoid_pick" in action_arg_name:
                    action_arg_value = (
                        np.array([random.randint(0, self.num_items_nav), 1])
                        + 1
                    )

                else:
                    action_arg_value = action_arg.sample()

                all_action_args[action_arg_name] = action_arg_value
        return {"action": tuple(all_actions), "action_args": all_action_args}


class HabDemoRunner:
    def __init__(self, args):
        self.args = args

    def step_env(self, action):
        start = time.time()
        obs = self.envs.step(action[0])  # type: ignore[has-type]
        step_time = time.time() - start
        return obs, step_time

    @staticmethod
    def _pool_init(b):
        global _barrier
        _barrier = b

    def do_time_steps(self):
        final_vid = []
        profile_sums = defaultdict(lambda: 0)  # type: ignore[var-annotated]

        for step_idx in range(self.args.n_steps):
            if self.envs.episode_over:
                break
            # print(f"step_idx = {step_idx}")
            actions = self.get_actions(step_idx)  # type: ignore[has-type]

            obs, step_time = self.step_env(actions)
            if step_idx >= self.args.n_pre_step:
                # Won't count the time of a pre-step
                profile_sums["time"] += step_time

            if self.args.render:
                for _ in range(self.args.n_procs):
                    final_vid.append(obs)

        if self.args.render and len(final_vid) > 0:
            from habitat_sim.utils import viz_utils as vut

            # TODO: setup an optional 3rd person render camera for debugging
            sensor_to_use = "head_rgb"
            if "agent_1_head_depth" not in final_vid[0]:
                sensor_to_use = "third_rgb"
            breakpoint()
            vut.make_video(
                final_vid,
                sensor_to_use,
                "color",
                "data/profile/big_benchmark_render_output",
                open_vid=True,
            )

        return dict(profile_sums)

    def _bench_target(self, _idx=0):
        self.init_common(_idx)

        if self.args.n_procs > 1 and _barrier is not None:
            _barrier.wait()
            if _idx == 0:
                _barrier.reset()
        profile_sums = self.do_time_steps()
        # self.envs.close()
        del self.envs  # type: ignore[has-type]

        return profile_sums

    def init_common(self, proc_idx):
        if self.args.n_gpus == 8:
            cores_per_proc = 8
        else:
            cores_per_proc = 16

        if platform != "darwin":
            # cpu_affinity only supported on linux/windows
            import psutil

            procs_per_gpu = args.n_procs // args.n_gpus
            gpu_idx = proc_idx // procs_per_gpu
            current_process = psutil.Process()
            orig_cpus = current_process.cpu_affinity()
            cpus = []
            for idx in range(len(orig_cpus) // 2):
                cpus.append(orig_cpus[idx])
                cpus.append(orig_cpus[idx + len(orig_cpus) // 2])

            current_process.cpu_affinity(
                cpus[gpu_idx * cores_per_proc : (gpu_idx + 1) * cores_per_proc]
            )

        self.envs = create_env(self.args, proc_idx)
        self.envs.reset()
        if hasattr(self.envs, "action_space"):
            ac_space = self.envs.action_space
        else:
            ac_space = self.envs.action_spaces[0]
        ac_space = ActionSpaceWrapper(ac_space.spaces)
        ac_space.num_items_nav = len(
            self.envs.task.pddl_problem.get_ordered_entities_list()
        )
        if self.args.load_actions is not None:
            with open(self.args.load_actions, "rb") as f:
                use_actions = np.load(f)
            if len(use_actions) != self.args.n_steps:
                raise ValueError(
                    "Loading action trajectory of size %i vs %i"
                    % (len(use_actions), self.args.n_steps)
                )
            # create an action dictionary compatible with loaded rearrange arm actions
            self.get_actions = lambda i: np.array(
                [
                    {
                        "action": "arm_action",
                        "action_args": {"arm_action": use_actions[i][:-1]},
                    }
                    for _ in range(self.args.n_procs)
                ]
            )

        else:
            self.get_actions = lambda i: np.array(
                [ac_space.sample(i) for _ in range(self.args.n_procs)]
            )

    def benchmark(self):
        if self.args.n_procs == 1:  # or self.args.vector_env:
            return self._bench_target()
        else:
            barrier = multiprocessing.Barrier(self.args.n_procs)
            with multiprocessing.Pool(
                self.args.n_procs,
                initializer=self._pool_init,
                initargs=(barrier,),
            ) as pool:
                perfs = pool.map(self._bench_target, range(self.args.n_procs))
            res = {k: 0 for k in perfs[0].keys()}
            for p in perfs:
                for k, v in p.items():
                    # Since we were running all the processes concurrently.
                    res[k] += v / args.n_procs
            return res


if __name__ == "__main__":
    load_fname = "orp/start_data/bench_ac.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", type=str, default="")
    parser.add_argument(
        "--cfg", type=str, default="benchmark/rearrange/idle.yaml"
    )

    parser.add_argument(
        "--n-procs",
        type=int,
        default=1,
        help="""
            Total number of processes. NOT number of processes per GPU.
            """,
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="""
            Number of GPUs to evenly spread --n-procs between.
            """,
    )
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--n-pre-step", type=int, default=10)
    parser.add_argument("--reset-interval", type=int, default=-1)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()

    fps_accumulator = []
    avg_fps = 0

    for _trial in range(args.n_trials):
        bench = HabDemoRunner(args)
        profile_sums = bench.benchmark()

        total_steps = (args.n_steps - args.n_pre_step) * args.n_procs

        fps = total_steps / profile_sums["time"]

        fps_accumulator.append(fps)
        avg_fps += fps

        profile_k = sorted(profile_sums.keys())
        profile_avgs = {k: profile_sums[k] / total_steps for k in profile_k}

        save_str = ""
        save_str += f"hab3: {args.n_procs} processes, {args.n_steps} steps with resets every {args.reset_interval} steps\n"
        save_str += f"FPS: {fps}\n"
        save_str += "Average time per function call (in seconds):\n"
        for k, v in profile_avgs.items():
            save_str += f"  - {k}: {v}s\n"

        print(save_str)
        scene_id = args.cfg.split("/")[-1].split(".")[0]
        save_dir = "data/profile/hab3"
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{save_dir}/{args.n_procs}_{args.n_steps}_{args.reset_interval}_{args.out_name}.txt"
        with open(fname, "w") as f:
            f.write(save_str)

        print("Wrote result to ", fname)

    avg_fps /= args.n_trials
    print("================================================================")
    print(
        f"Ran {args.n_trials} trial(s) with average FPS of {avg_fps} from {fps_accumulator}."
    )
    print("================================================================")
    # breakpoint()
