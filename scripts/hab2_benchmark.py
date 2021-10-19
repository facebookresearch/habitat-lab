import argparse
import multiprocessing
import os
import os.path as osp
import time
from collections import defaultdict

import cv2
import numpy as np

import habitat


def save_mp4(frames, vid_dir, name, fps=60.0, no_frame_drop=False):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    vid_file = osp.join(vid_dir, name + ".mp4")
    if osp.exists(vid_file):
        os.remove(vid_file)

    w, h = frames[0].shape[:-1]
    videodims = (h, w)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(vid_file, fourcc, fps, videodims)
    for frame in frames:
        frame = frame[..., 0:3][..., ::-1]
        video.write(frame)
    video.release()
    print(f"Rendered to {vid_file}")


def create_env(args, proc_i):
    procs_per_gpu = args.n_procs // args.n_gpus
    procs_to_gpu = {i: i // procs_per_gpu for i in range(args.n_procs)}
    spec_gpu = procs_to_gpu[proc_i]
    print(f"assigning {proc_i} gpu {spec_gpu}")

    set_opts = args.opts
    set_opts.extend(["SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID", spec_gpu])

    config = habitat.get_config(args.cfg, set_opts)
    return habitat.Env(config=config)


_barrier = None


class HabDemoRunner:
    def __init__(self, args):
        self.args = args

    def step_env(self, action):
        start = time.time()
        obs = self.envs.step(action[0])
        step_time = time.time() - start

        return obs, step_time

    @staticmethod
    def _pool_init(b):
        global _barrier
        _barrier = b

    def do_time_steps(self):
        final_vid = []
        profile_sums = defaultdict(lambda: 0)

        for step_idx in range(self.args.n_steps):
            actions = self.get_actions(step_idx)

            obs, step_time = self.step_env(actions)
            if step_idx >= self.args.n_pre_step:
                # Won't count the time of a pre-step
                profile_sums["time"] += step_time

            if self.args.render:
                for i in range(self.args.n_procs):
                    final_vid.append(obs)

        if self.args.render and len(final_vid) > 0:
            from habitat_sim.utils import viz_utils as vut
            #TODO: setup an optional 3rd person render camera for debugging
            vut.make_video(
                final_vid,
                "robot_head_rgb",
                "color",
                "benchmark_render_output",
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
        del self.envs

        return profile_sums

    def init_common(self, proc_idx):
        self.envs = create_env(self.args, proc_idx)
        self.envs.reset()
        if hasattr(self.envs, "action_space"):
            ac_space = self.envs.action_space
        else:
            ac_space = self.envs.action_spaces[0]
        if self.args.load_actions is not None:
            with open(self.args.load_actions, "rb") as f:
                use_actions = np.load(f)
            if len(use_actions) != self.args.n_steps:
                raise ValueError(
                    "Loading action trajectory of size %i vs %i"
                    % (len(use_actions), self.args.n_steps)
                )
            self.get_actions = lambda i: np.array(
                [use_actions[i] for _ in range(self.args.n_procs)]
            )

        else:
            self.get_actions = lambda i: np.array(
                [ac_space.sample() for _ in range(self.args.n_procs)]
            )

    def benchmark(self):
        if self.args.n_procs == 1:# or self.args.vector_env:
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
    parser.add_argument("--cfg", type=str, default="")

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
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--n-pre-step", type=int, default=1)
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

    final_vid = []
    bench = HabDemoRunner(args)
    profile_sums = bench.benchmark()

    total_steps = (args.n_steps - args.n_pre_step) * args.n_procs

    fps = total_steps / profile_sums["time"]

    profile_k = sorted(profile_sums.keys())
    profile_avgs = {k: profile_sums[k] / total_steps for k in profile_k}

    save_str = ""
    save_str += f"hab2: {args.n_procs} processes, {args.n_steps} steps with resets every {args.reset_interval} steps\n"
    save_str += f"FPS: {fps}\n"
    save_str += "Average time per function call (in seconds):\n"
    for k, v in profile_avgs.items():
        save_str += f"  - {k}: {v}s\n"

    print(save_str)
    scene_id = args.cfg.split("/")[-1].split(".")[0]
    save_dir = "data/profile"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/{args.n_procs}_{args.n_steps}_{args.reset_interval}_{scene_id}_{args.out_name}.txt"
    with open(fname, "w") as f:
        f.write(save_str)

    print("Wrote result to ", fname)
