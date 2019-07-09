import os
import time
from collections import defaultdict, deque

import torch

from habitat import logger
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)
from habitat_baselines.common.base_model import BaseRLModel
from habitat_baselines.common.env_utils import NavRLEnv, construct_envs
from habitat_baselines.common.tensorboard_utils import get_tensorboard_writer
from habitat_baselines.common.trainer_registry import train_registry
from habitat_baselines.common.utils import (
    _flatten_helper,
    batch_obs,
    poll_checkpoint_folder,
    update_linear_schedule,
)
from habitat_baselines.rl.ppo import PPO, Policy


@train_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLModel):
    def __init__(self, config=None):
        super().__init__(config)
        print(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.device = None
        if config is not None:
            logger.info(f"env config: {config}")
            self.envs = construct_envs(config, NavRLEnv)
            self._setup()

    def _setup(self):
        ppo_cfg = self.config.BASELINE.RL.PPO
        self.device = torch.device("cuda", ppo_cfg.pth_gpu_id)
        logger.add_filehandler(ppo_cfg.log_file)

        self.actor_critic = Policy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=512,
            goal_sensor_uuid=self.config.TASK.GOAL_SENSOR_UUID,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            self.actor_critic,
            ppo_cfg.clip_param,
            ppo_cfg.ppo_epoch,
            ppo_cfg.num_mini_batch,
            ppo_cfg.value_loss_coef,
            ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

    def save_checkpoint(self, file_name: str):
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config_str": str(self.config),
        }
        torch.save(
            checkpoint,
            os.path.join(
                self.config.BASELINE.RL.PPO.checkpoint_folder, file_name
            ),
        )

    def load_checkpoint(self):
        pass

    def train(self):
        assert (
            self.config is not None
        ), "trainer is not properly initialized, need to specify config file"

        ppo_cfg = self.config.BASELINE.RL.PPO
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        observations = self.envs.reset()

        batch = batch_obs(observations)

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        rollouts.to(self.device)

        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        with (
            get_tensorboard_writer(
                log_dir=ppo_cfg.tensorboard_dir,
                purge_step=count_steps,
                flush_secs=30,
            )
        ) as writer:
            for update in range(ppo_cfg.num_updates):
                if ppo_cfg.use_linear_lr_decay:
                    update_linear_schedule(
                        self.agent.optimizer,
                        update,
                        ppo_cfg.num_updates,
                        ppo_cfg.lr,
                    )

                self.agent.clip_param = ppo_cfg.clip_param * (
                    1 - update / ppo_cfg.num_updates
                )

                for step in range(ppo_cfg.num_steps):
                    t_sample_action = time.time()
                    # sample actions
                    with torch.no_grad():
                        step_observation = {
                            k: v[step]
                            for k, v in rollouts.observations.items()
                        }

                        (
                            values,
                            actions,
                            actions_log_probs,
                            recurrent_hidden_states,
                        ) = self.actor_critic.act(
                            step_observation,
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step],
                        )
                    pth_time += time.time() - t_sample_action

                    t_step_env = time.time()

                    outputs = self.envs.step([a[0].item() for a in actions])
                    observations, rewards, dones, infos = [
                        list(x) for x in zip(*outputs)
                    ]

                    env_time += time.time() - t_step_env

                    t_update_stats = time.time()
                    batch = batch_obs(observations)
                    rewards = torch.tensor(rewards, dtype=torch.float)
                    rewards = rewards.unsqueeze(1)

                    masks = torch.tensor(
                        [[0.0] if done else [1.0] for done in dones],
                        dtype=torch.float,
                    )

                    current_episode_reward += rewards
                    episode_rewards += (1 - masks) * current_episode_reward
                    episode_counts += 1 - masks
                    current_episode_reward *= masks

                    rollouts.insert(
                        batch,
                        recurrent_hidden_states,
                        actions,
                        actions_log_probs,
                        values,
                        rewards,
                        masks,
                    )

                    count_steps += self.envs.num_envs
                    pth_time += time.time() - t_update_stats

                window_episode_reward.append(episode_rewards.clone())
                window_episode_counts.append(episode_counts.clone())

                t_update_model = time.time()
                with torch.no_grad():
                    last_observation = {
                        k: v[-1] for k, v in rollouts.observations.items()
                    }
                    next_value = self.actor_critic.get_value(
                        last_observation,
                        rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1],
                    ).detach()

                rollouts.compute_returns(
                    next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
                )

                value_loss, action_loss, dist_entropy = self.agent.update(
                    rollouts
                )

                rollouts.after_update()
                pth_time += time.time() - t_update_model

                losses = [value_loss, action_loss]
                stats = zip(
                    ["count", "reward"],
                    [window_episode_counts, window_episode_reward],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % ppo_cfg.log_interval == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % ppo_cfg.checkpoint_interval == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

    def eval(self):
        ppo_cfg = self.config.BASELINE.RL.PPO

        assert (len(ppo_cfg.model_path) > 0) != (
            len(ppo_cfg.tracking_model_dir) > 0
        ), "Must specify a single model or a directory of models, but not both"
        if "tensorboard" in ppo_cfg.video_option:
            assert (
                ppo_cfg.tensorboard_dir is not None
            ), "Must specify a tensorboard directory for video display"
        if "disk" in ppo_cfg.video_option:
            assert (
                ppo_cfg.video_dir is not None
            ), "Must specify a directory for storing videos on disk"

        with get_tensorboard_writer(
            ppo_cfg.tensorboard_dir, purge_step=0, flush_secs=30
        ) as writer:
            if len(ppo_cfg.model_path) > 0:
                # evaluate singe checkpoint
                self._eval_checkpoint(ppo_cfg.model_path, ppo_cfg, writer)
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            ppo_cfg.tracking_model_dir, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.warning(
                        "=============current_ckpt: {}=============".format(
                            current_ckpt
                        )
                    )
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        current_ckpt,
                        ppo_cfg,
                        writer,
                        cur_ckpt_idx=prev_ckpt_ind,
                    )

    def _eval_checkpoint(self, checkpoint_path, args, writer, cur_ckpt_idx=0):
        device = torch.device("cuda", args.pth_gpu_id)
        config = self.config.clone()
        config.defrost()
        config.DATASET.SPLIT = "val"
        agent_sensors = args.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        config.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        if args.video_option:
            config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, NavRLEnv)

        ckpt = torch.load(checkpoint_path, map_location=device)

        actor_critic = Policy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=512,
            goal_sensor_uuid=config.TASK.GOAL_SENSOR_UUID,
        )
        actor_critic.to(device)

        # TODO read ppo params from saved config
        ppo = PPO(
            actor_critic=actor_critic,
            clip_param=0.1,
            ppo_epoch=4,
            num_mini_batch=32,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            lr=2.5e-4,
            eps=1e-5,
            max_grad_norm=0.5,
        )

        ppo.load_state_dict(ckpt["state_dict"])

        actor_critic = ppo.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=device
        )

        test_recurrent_hidden_states = torch.zeros(
            args.num_processes, args.hidden_size, device=device
        )
        not_done_masks = torch.zeros(args.num_processes, 1, device=device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = None
        if args.video_option:
            rgb_frames = [[]] * args.num_processes
            os.makedirs(args.video_dir, exist_ok=True)

        while (
            len(stats_episodes) < args.count_test_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    deterministic=False,
                )

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats["spl"] = infos[i]["spl"]
                    episode_stats["success"] = int(infos[i]["spl"] > 0)
                    episode_stats["reward"] = current_episode_reward[i].item()
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    if args.video_option:
                        generate_video(
                            args,
                            rgb_frames[i],
                            current_episodes[i].episode_id,
                            cur_ckpt_idx,
                            infos[i]["spl"],
                            writer,
                        )
                        rgb_frames[i] = []

                # episode continues
                elif args.video_option:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            # pausing self.envs with no new episode
            if len(envs_to_pause) > 0:
                state_index = list(range(self.envs.num_envs))
                for idx in reversed(envs_to_pause):
                    state_index.pop(idx)
                    self.envs.pause_at(idx)

                # indexing along the batch dimensions
                test_recurrent_hidden_states = test_recurrent_hidden_states[
                    state_index
                ]
                not_done_masks = not_done_masks[state_index]
                current_episode_reward = current_episode_reward[state_index]

                for k, v in batch.items():
                    batch[k] = v[state_index]

                if args.video_option:
                    rgb_frames = [rgb_frames[i] for i in state_index]

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_spl_mean = aggregated_stats["spl"] / num_episodes
        episode_success_mean = aggregated_stats["success"] / num_episodes

        logger.info(
            "Average episode reward: {:.6f}".format(episode_reward_mean)
        )
        logger.info(
            "Average episode success: {:.6f}".format(episode_success_mean)
        )
        logger.info("Average episode SPL: {:.6f}".format(episode_spl_mean))

        writer.add_scalars(
            "eval_reward",
            {"average reward": episode_reward_mean},
            cur_ckpt_idx,
        )
        writer.add_scalars(
            "eval_SPL", {"average SPL": episode_spl_mean}, cur_ckpt_idx
        )
        writer.add_scalars(
            "eval_success",
            {"average success": episode_success_mean},
            cur_ckpt_idx,
        )


def generate_video(
    ppo_cfg, images, episode_id, checkpoint_idx, spl, tb_writer, fps=10
) -> None:
    r"""Generate video according to specified information.

    ppo_cfg:
        ppo_cfg: contains ppo_cfg.video_option and ppo_cfg.video_dir.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        spl: SPL for this episode for video naming.
        tb_writer: tensorboard writer object for uploading video
        fps: fps for generated video

    Returns:
        None
    """
    if ppo_cfg.video_option and len(images) > 0:
        video_name = f"episode{episode_id}_ckpt{checkpoint_idx}_spl{spl:.2f}"
        if "disk" in ppo_cfg.video_option:
            images_to_video(images, ppo_cfg.video_dir, video_name)
        if "tensorboard" in ppo_cfg.video_option:
            tb_writer.add_video_from_np_images(
                f"episode{episode_id}", checkpoint_idx, images, fps=fps
            )


class RolloutStorage:
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape,
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_envs, recurrent_hidden_state_size
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )

                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind]
                )

                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = _flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )
