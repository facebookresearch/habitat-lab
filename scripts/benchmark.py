import numpy as np
import argparse
from tqdm import tqdm
import os.path as osp
import os
import time
import logging
import cv2
from gym import spaces
from collections import defaultdict
import multiprocessing

try:
    from gibson2.envs.igibson_env import iGibsonEnv
    from gibson2.envs.parallel_env import ParallelNavEnv
    from gibson2.simulator import Simulator
    from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
    from gibson2.robots.turtlebot_robot import Turtlebot
    from gibson2.robots.fetch_robot import Fetch
    from gibson2.utils.constants import NamedRenderingPresets
    from gibson2.utils.utils import parse_config
    from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
    import gibson2
except:
    print('Not loading igibson')


try:
    #from orp_env_adapter import get_hab_envs, get_hab_args, OrpWrapper
    #from orp.env import get_env
    import habitat
    from habitat_baselines.config.default import get_config
    from habitat_baselines.utils.env_utils import construct_envs, make_env_fn
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    import habitat_sim
except:
    print('Not loading Habitat')

try:
    import ravens
except ImportError:
    print('Not loading Ravens')


def save_mp4(frames, vid_dir, name, fps=60.0, no_frame_drop=False):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    vid_file = osp.join(vid_dir, name + '.mp4')
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


class BareHab:
    def __init__(self, scene_name, config, spec_gpu, overrides):
        #combined_settings = ','.join([f"{k}={v}" for k,v in settings])
        #TODO: verify this change
        #self.sim = get_env(scene_name, agent_config, env_config, 'train',
        #        overrides=overrides, spec_gpu=spec_gpu)
        config = get_config("habitat_baselines/config/rearrange/ddppo_rearrangepick.yaml") 
        from habitat_baselines.common.environments import get_env_class
        env_class = get_env_class(config.ENV_NAME)
        self.sim = make_env_fn(env_class=env_class, config=config)
        #end TODO:
        ac_space = spaces.Box(shape=(4,), low=0, high=1, dtype=np.float32)
        self.action_spaces = [ac_space]
        #NOTE: we need to reset the env before doing any steps
        self.reset()

    def step(self, a):
        action = self.sim.action_space.sample()
        obs = self.sim.step(action=action)
        #obs = self.sim.step('EMPTY', {'empty': a})
        ep_profile = {}
        for log_timer in self.sim._env._sim.timer.timers:
            time_val, call_count = self.sim._env._sim.timer.get_time(log_timer)
            ep_profile[log_timer] = time_val / call_count if call_count != 0 else 0
        return [(obs, 0, 0, {'profile': ep_profile})]

    def close(self):
        self.sim.close()

    def reset(self):
        return self.sim.reset()


class BareIGib:
    def __init__(self, scene_name, parallel):
        config = parse_config('orp/start_data/igibson_fetch.yaml')
        scene = InteractiveIndoorScene(
            scene_name, texture_randomization=False, object_randomization=False)
        settings = MeshRendererSettings(
            msaa=False, enable_shadow=False, optimized=True)
        s = Simulator(mode='headless',
                      image_width=512,
                      image_height=512,
                      device_idx=0,
                      rendering_settings=settings,
                      )
        s.import_ig_scene(scene)
        self.robot = Fetch(config)
        s.import_robot(self.robot)

        s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
        self.s = s

        self.action_space = self.robot.action_space
        self.observation_space = spaces.Dict({
                'rgb': spaces.Box(shape=(128,128,3), low=0.0,high=1.0)})
        self.parallel = parallel

    def step(self,a):
        self.s.step()
        self.robot.apply_action(a[0])
        obs = self.s.renderer.render_robot_cameras(modes=('rgb'))
        if self.parallel:
            return obs, 0.0, 0.0, {'profile': {}}
        else:
            return [(obs, None, 0.0, {'profile': {}})]

    def reset(self):
        #for _ in range(10):
        #    self.step(None)
        return None

class EnvRaven:
    def __init__(self, scene_name):
        self.env = ravens.Environment(
                'ravens/ravens/environments/assets/',
                disp=False,
                shared_memory=False,
                hz=480)
        task = ravens.tasks.names[args.scene_name]()
        task.mode = 'train'
        self.task = task
        self.agent = task.oracle(self.env)
        self.env.set_task(task)


    def step(self, a):
        obs, reward, done, info = self.env.step(a[0])
        return [(obs, reward, done, info)]

    def reset(self):
        self.env.set_task(self.task)
        return [self.env.reset()]

    @property
    def action_space(self):
        return self.env.action_space

class EnvIGib:
    def __init__(self, scene_name):
        self.env = iGibsonEnv(config_file='orp/start_data/igibson_fetch.yaml',
                action_timestep=1 / 30.0,
                physics_timestep=1 / 120.0,
                mode='headless', scene_id=scene_name)
        self.action_space = self.env.action_space
    def step(self, a):
        obs, reward, done, info = self.env.step(a[0])
        return [(obs, reward, done, info)]

    def reset(self):
        return [self.env.reset()]

def make_hab_env(scene_name, overrides, spec_gpu):
    tmp = BareHab(scene_name, None, spec_gpu, overrides)
    return tmp.sim

def create_env(args, proc_i):
    if args.raven:
        envs = EnvRaven(args.scene_name)
    elif not args.igib:
        procs_per_gpu = args.n_procs // args.n_gpus
        procs_to_gpu = {i: i // procs_per_gpu for i in range(args.n_procs)}
        spec_gpu = procs_to_gpu[proc_i]
        print(f"assigning {proc_i} gpu {spec_gpu}")
        if args.full_task:
            env_args = ['hab_scene_name', args.scene_name,
                    'NUM_PROCESSES', args.n_procs,
                    'NUM_ENVIRONMENTS', args.n_procs,
                    'hab_env_config', 'empty']
            env_args.extend(['hab_agent_config', args.agent_config])
            envs, _ = get_hab_envs(get_config('./method/cfgs/ppo_cfg.yaml', env_args),
                    'config.yaml', False, ret_no_vec_env=not args.vector_env,
                    spec_gpu=spec_gpu)
        else:
            if args.n_procs > 1 and args.vector_env:
                env_args = [tuple([args.scene_name, args.override, spec_gpu])
                        for i in range(args.n_procs)]

                envs = habitat.VectorEnv(
                        make_env_fn=make_hab_env,
                        env_fn_args=env_args)
            else:
                envs = BareHab(args.scene_name, None,
                        spec_gpu, args.override)
    else:
        def load_env():
            return BareIGib(args.scene_name, True)

        if args.n_procs > 1 and args.vector_env:
            envs = ParallelNavEnv([load_env] * args.n_procs, blocking=False)
        else:
            envs = BareIGib(args.scene_name, False)
    if not args.full_task:
        obs = envs.reset()
    return envs


_barrier = None

class HabDemoRunner:
    def __init__(self, args):
        self.args = args

    def step_env(self, action):
        if not self.args.igib and not self.args.raven:
            if self.args.full_task:
                if self.args.vector_env:
                    step_data = [{'action': a} for a in action]
                    start = time.time()
                    outputs = self.envs.step(step_data[0])
                    step_time = time.time() - start
                else:
                    start = time.time()
                    obs, reward, done, info = self.envs.step(action[0])
                    step_time = time.time() - start
                    return [obs], [reward], [done], [info], step_time
            else:
                use_ac = [{ 'action_name': 'EMPTY',
                            'action_args': ac
                            } for ac in action]

                start = time.time()
                outputs = self.envs.step(use_ac)
                step_time = time.time() - start

            obs, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            if not self.args.full_task and self.args.n_procs == 1:
                for i in range(self.args.n_procs):
                    obs[i] = obs[i][0]
            return obs, rewards_l, dones, infos, step_time
        else:
            start = time.time()
            res = self.envs.step(action)
            step_time = time.time() - start

            obs = []
            reward = []
            done = []
            info = []
            for (ind_obs,ind_reward,ind_done,ind_info) in res:
                obs.append(ind_obs)
                reward.append(ind_reward)
                done.append(ind_done)
                info.append(ind_info)
            return obs, reward, done, info, step_time

    @staticmethod
    def _pool_init(b):
        global _barrier
        _barrier = b

    def do_time_steps(self):
        final_vid = []
        profile_sums = defaultdict(lambda: 0)

        for step_idx in range(self.args.n_steps):
            actions = self.get_actions(step_idx)
            if not self.args.vector_env:
                actions = actions[:1]

            obs, reward, done, info, step_time = self.step_env(actions)
            if step_idx >= self.args.n_pre_step:
                # Won't count the time of a pre-step
                profile_sums['time'] += step_time

            # Check there are no accidental resets occuring in step_env causing a
            # slow down.
            assert sum(done) == 0

            if self.args.vector_env:
                use_range = range(self.args.n_procs)
            else:
                use_range = [0]

            for i in use_range:
                if 'profile' not in info[i]:
                    continue
                for k,v in info[i]['profile'].items():
                    profile_sums[k] += v / len(use_range)

            if self.args.render:
                for i in range(self.args.n_procs):
                    final_vid.append(obs[i]['rgb'])

        if self.args.render and len(final_vid) > 0:
            if final_vid[0].dtype == np.float32:
                for i in range(len(final_vid)):
                    final_vid[i] *= 255.0
                    final_vid[i] = final_vid[i].astype(np.uint8)
            save_mp4(final_vid, './data/vids', 'bench')
            print(f"Saved video to data/vids/bench.mp4")

        return dict(profile_sums)

    def _bench_target(self, _idx=0):
        self.init_common(_idx)

        if self.args.n_procs > 1 and not self.args.vector_env and _barrier is not None:
            _barrier.wait()
            if _idx == 0:
                _barrier.reset()
        profile_sums = self.do_time_steps()
        #self.envs.close()
        del self.envs

        return profile_sums

    def init_common(self, proc_idx):
        self.envs = create_env(self.args, proc_idx)
        self.envs.reset()
        if hasattr(self.envs, 'action_space'):
            ac_space = self.envs.action_space
        else:
            ac_space = self.envs.action_spaces[0]
        if self.args.load_actions is not None:
            with open(self.args.load_actions, 'rb') as f:
                use_actions = np.load(f)
            if len(use_actions) != self.args.n_steps:
                raise ValueError('Loading action trajectory of size %i vs %i' % (len(use_actions), self.args.n_steps))
            self.get_actions = lambda i: np.array([use_actions[i] for _ in
                range(self.args.n_procs)])

        else:
            self.get_actions = lambda i: np.array([ac_space.sample() for _ in
                range(self.args.n_procs)])

    def benchmark(self):
        if self.args.n_procs == 1 or self.args.vector_env:
            return self._bench_target()
        else:
            barrier = multiprocessing.Barrier(self.args.n_procs)
            with multiprocessing.Pool(
                self.args.n_procs, initializer=self._pool_init, initargs=(barrier,)
            ) as pool:
                perfs = pool.map(self._bench_target, range(self.args.n_procs))
            res = {k: 0 for k in perfs[0].keys()}
            for p in perfs:
                for k, v in p.items():
                    # Since we were running all the processes concurrently.
                    res[k] += v / args.n_procs
            return res



if __name__ == '__main__':
    #load_fname = 'orp/start_data/bench_ac.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-name', type=str, default='bench')
    parser.add_argument('--out-name', type=str, default='')
    parser.add_argument('--override', type=str, default='')

    parser.add_argument('--n-procs', type=int, default=1, help="""
            Total number of processes. NOT number of processes per GPU.
            """)
    parser.add_argument('--n-gpus', type=int, default=1, help="""
            Number of GPUs to evenly spread --n-procs between.
            """)
    parser.add_argument('--vector-env', action='store_true', help="""
            Use the action synchronized multiprocess code. You probably
            shouldn't do this.
            """)
    parser.add_argument('--n-steps', type=int, default=10000)
    parser.add_argument('--n-pre-step', type=int, default=1)
    parser.add_argument('--reset-interval', type=int, default=-1)

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--agent-config', type=str, default='empty')
    parser.add_argument('--load-actions', type=str, default=None)
    parser.add_argument('--full-task', action='store_true')

    # iGibson related arguments.
    parser.add_argument('--disable-igib-log', action='store_true')
    parser.add_argument('--igib', action='store_true')
    parser.add_argument('--raven', action='store_true')
    args = parser.parse_args()

    if args.disable_igib_log:
        logging.disable(logging.CRITICAL)

    final_vid = []
    bench= HabDemoRunner(args)
    profile_sums = bench.benchmark()

    total_steps = (args.n_steps - args.n_pre_step) * args.n_procs

    fps = total_steps / profile_sums['time']

    profile_k = sorted(list(profile_sums.keys()))
    profile_avgs = {k: profile_sums[k] / total_steps for k in profile_k}

    save_str = ""
    if args.igib:
        ident_str = "igib"
    else:
        ident_str = "hab"

    save_str += f"{ident_str}: {args.n_procs} processes, {args.n_steps} steps with resets every {args.reset_interval} steps\n"
    save_str += f"FPS: {fps}\n"
    save_str += "Average time per function call (in seconds):\n"
    for k,v in profile_avgs.items():
        save_str += f"  - {k}: {v}s\n"

    print(save_str)
    scene_id = args.scene_name.replace('/', '_')
    fname = f"data/profile/{ident_str}_{args.n_procs}_{args.n_steps}_{args.reset_interval}_{scene_id}_{args.out_name}.txt"
    with open(fname, 'w') as f:
        f.write(save_str)

    print('Wrote result to ', fname)


