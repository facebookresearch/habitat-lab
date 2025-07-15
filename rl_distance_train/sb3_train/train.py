import os
import torch
import gym
import habitat.gym                    # registers Habitat envs in Gym
from habitat import get_config
from habitat.gym.gym_definitions import make_gym_from_config
from habitat.config import read_write

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
)

from utils import SyncEvalEnvStatsCallback

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)



# 1. Env factory
def make_env():
    cfg = get_config(
        config_path=os.path.join(
            "benchmark", "nav", "pointnav", "pointnav_gibson.yaml"
        ),
    )
    return make_gym_from_config(cfg)

def make_eval_env():
    cfg = get_config(
        config_path=os.path.join(
            "benchmark", "nav", "pointnav", "pointnav_gibson.yaml"
        ),
    )

    with read_write(cfg):
        cfg.habitat.dataset.split = "val"

    return make_gym_from_config(cfg)

# 2. Vectorized + normalized env
num_train_envs = 2
train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

# 3. Set device to “cuda” so SB3 puts the policy on GPU
#    (“auto” would pick cuda:0 if available, but we want explicit torch.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PPO(
    policy="MultiInputPolicy",
    env=train_env,
    verbose=2,
    tensorboard_log="tb/sb3_ppo_habitat/",
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    device=device,
)

# 4. Train with live tqdm-style output
total_timesteps = 1_000_000
sync_stats_callback = SyncEvalEnvStatsCallback(train_env, eval_env, sync_freq=2048 * num_train_envs, verbose=0)
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path=f"data/sb3_ppo_habitat/",
    log_path=f"tb/sb3_ppo_habitat/",
    eval_freq=2048 * num_train_envs,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1,
)

callbacks = CallbackList([sync_stats_callback, eval_callback])
model.learn(
    total_timesteps=int(total_timesteps),
    callback=callbacks,
    progress_bar=True,
    log_interval=100,
)

# 5. Save artifacts
# model.save("ppo_habitat_pointnav_gibson")
# env.save("vecnormalize_pointnav_gibson.pkl")


# 6. (Optional) evaluation loop
# --------------------------------
# from stable_baselines3.common.vec_env import VecEnvWrapper
# # reload env & stats
# eval_env = DummyVecEnv([make_env])
# eval_env = VecNormalize.load("vecnormalize_pointnav_gibson.pkl", eval_env)
# eval_env.training = False  # turn off norm updating
# obs = eval_env.reset()
# for _ in range(500):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = eval_env.step(action)
#     eval_env.render()
#     if done:
#         obs = eval_env.reset()
