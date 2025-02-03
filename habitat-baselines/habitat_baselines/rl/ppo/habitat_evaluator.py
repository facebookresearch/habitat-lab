import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from einops import rearrange
from PIL import Image

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info


class HabitatEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    observation_dict: List[Any] = (
        []
    )  # This is to store the past observations from habitat
    vla_action: List[Any] = (
        []
    )  # This is to store the actions from action chunk
    depoly_one_action = True  # If we want to do MPC style -- only depoly one action from action chunk
    # TODO: expand these to multi-sensors
    vla_target_image = "articulated_agent_arm_rgb"  # Target RGB
    vla_target_proprio = "joint"  # Target proprio sensor
    # vla_target_proprio = "ee_pos"  # Target proprio sensor

    @staticmethod
    def process_rgb(rgbs, target_size):
        """Resize the rgb images"""
        # Resize the image here
        rgbs_process = torch.zeros(
            (rgbs.shape[0], 3, target_size, target_size)
        )
        for i, rgb in enumerate(rgbs):
            img = Image.fromarray(rgb.cpu().detach().numpy())
            img = img.resize((target_size, target_size))
            img = np.array(img)
            rgb = torch.as_tensor(
                rearrange(img, "h w c-> c h w")
            )  # torch.Size([3, 224, 224])
            rgbs_process[i] = rgb
        return rgbs_process

    def infer_action_vla_model(
        self,
        vla_model,
        processor,
        observation,
        device,
        vla_config,
        current_episodes_info,
    ):
        """Infer action using vla models."""
        self.observation_dict.append(observation)
        # Confirm the number of batches
        bsz = observation[self.vla_target_image].shape[0]
        dummy_images = torch.randint(
            0,
            256,
            (
                bsz,
                vla_config.cond_steps,
                3,
                vla_config.image_size,
                vla_config.image_size,
            ),
            dtype=torch.uint8,
        )
        dummy_proprio = torch.zeros(
            (bsz, vla_config.cond_steps, vla_config.proprio_dim)
        )
        if len(self.observation_dict) < vla_config.cond_steps:
            store_size = len(self.observation_dict)
            for i in range(store_size):
                dummy_images[:, vla_config.cond_steps - i - 1] = (
                    self.process_rgb(
                        self.observation_dict[-i - 1][self.vla_target_image],
                        vla_config.image_size,
                    )
                )
                if self.vla_target_proprio == "ee_pos":
                    prop_obs = self.observation_dict[-i - 1]["ee_pose"][:, :3]
                else:
                    prop_obs = self.observation_dict[-i - 1][
                        self.vla_target_proprio
                    ]
                dummy_proprio[:, vla_config.cond_steps - i - 1] = prop_obs
            # Pad the image one with the last image
            for i in range(vla_config.cond_steps - store_size):
                dummy_images[:, i] = self.process_rgb(
                    self.observation_dict[0][self.vla_target_image],
                    vla_config.image_size,
                )
                if self.vla_target_proprio == "ee_pos":
                    prop_obs = self.observation_dict[-i - 1]["ee_pose"][:, :3]
                else:
                    prop_obs = self.observation_dict[-i - 1][
                        self.vla_target_proprio
                    ]

                dummy_proprio[:, i] = prop_obs
        else:
            for i in range(vla_config.cond_steps):
                dummy_images[:, i] = self.process_rgb(
                    self.observation_dict[i - vla_config.cond_steps][
                        self.vla_target_image
                    ],
                    vla_config.image_size,
                )
                if self.vla_target_proprio == "ee_pos":
                    prop_obs = self.observation_dict[
                        i - vla_config.cond_steps
                    ]["ee_pose"][:, :3]
                else:
                    prop_obs = self.observation_dict[
                        i - vla_config.cond_steps
                    ][self.vla_target_proprio]

                dummy_proprio[:, i] = prop_obs

        dummy_images = rearrange(dummy_images, "B T C H W -> (B T) C H W")

        # TODO: get the text
        dummy_texts = [
            ep_info.language_instruction for ep_info in current_episodes_info
        ]

        dtype = torch.bfloat16
        # process image and text
        model_inputs = processor(text=dummy_texts, images=dummy_images)
        model_inputs["pixel_values"] = rearrange(
            model_inputs["pixel_values"],
            "(B T) C H W -> B T C H W",
            B=bsz,
            T=vla_config.cond_steps,
        )
        (
            causal_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
        ) = vla_model.build_causal_mask_and_position_ids(
            model_inputs["attention_mask"], dtype=dtype
        )
        (
            image_text_proprio_mask,
            action_mask,
        ) = vla_model.split_full_mask_into_submasks(causal_mask)

        with torch.inference_mode():
            actions = vla_model.infer_action(
                input_ids=model_inputs["input_ids"].to(device),
                pixel_values=model_inputs["pixel_values"].to(dtype).to(device),
                image_text_proprio_mask=image_text_proprio_mask.to(device),
                action_mask=action_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
            )  # [bsz, horizon, dim]
        return actions

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
        vla_model=None,
        vla_processor=None,
        vla_config=None,
    ):
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = (
            {}
        )  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            if (
                self.vla_action == [] or self.depoly_one_action
            ) and config.habitat_baselines.load_third_party_ckpt:
                self.vla_action = []
                vla_action = self.infer_action_vla_model(
                    vla_model,
                    vla_processor,
                    batch,
                    device,
                    vla_config,
                    current_episodes_info,
                )
                # Make the time horizon as a leading dimension
                vla_action = rearrange(vla_action, "b h a-> h b a")
                for vla_a_time in vla_action:
                    vla_temp = []
                    for vla_a_batch in vla_a_time:
                        vla_temp.append(
                            vla_a_batch.cpu().detach().float().numpy()
                        )
                    self.vla_action.append(vla_temp)

            a_action = self.vla_action.pop(0)
            # a_action = np.mean(np.array(self.vla_action), axis=0)
            # Depoly an action
            if config.habitat_baselines.load_third_party_ckpt:
                outputs = envs.step(a_action)
            else:
                outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )

                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    aggregated_stats = {}
                    print_ks = {
                        "place_success",
                        "pick_success",
                        "num_steps",
                        "ee_to_object_distance.0",
                        "robot_collisions.total_collisions",
                    }
                    for stat_key in print_ks:
                        aggregated_stats[stat_key] = np.mean(
                            [
                                v[stat_key]
                                for v in stats_episodes.values()
                                if stat_key in v
                            ]
                        )
                    num_evaled = len(stats_episodes.values())

                    print(
                        f"Evaluated {num_evaled} episodes. Average episode "
                        + "; ".join(
                            [
                                f" {k}: {v:.4f}"
                                for k, v in aggregated_stats.items()
                            ]
                        )
                    )

                    if "results_dir" in config.habitat_baselines:
                        results_dir = config.habitat_baselines.results_dir
                        success_key = next(
                            (
                                key
                                for key in episode_stats.keys()
                                if "success" in key
                            ),
                            None,
                        )
                        if results_dir != "" and success_key is not None:
                            if not os.path.isdir(results_dir):
                                os.makedirs(results_dir)
                            episode_steps_filename = os.path.join(
                                results_dir, f"ckpt_{checkpoint_index}.csv"
                            )
                            if not os.path.isfile(episode_steps_filename):
                                episode_steps_data = f"#,id,reward,dist2goal,total_collisions,{success_key},num_steps\n"
                            else:
                                with open(episode_steps_filename) as f:
                                    episode_steps_data = f.read()
                            try:
                                episode_steps_data += (
                                    "{},{},{},{},{},{},{}\n".format(
                                        num_evaled,
                                        current_episodes_info[i].episode_id,
                                        np.round(episode_stats["reward"], 2),
                                        np.round(
                                            episode_stats[
                                                "ee_to_object_distance.0"
                                            ],
                                            2,
                                        ),
                                        episode_stats[
                                            "robot_collisions.total_collisions"
                                        ],
                                        episode_stats[success_key],
                                        episode_stats["num_steps"],
                                    )
                                )  # number of steps taken
                                lines = episode_steps_data.split("\n")
                                with open(episode_steps_filename, "w") as f:
                                    f.write(episode_steps_data)
                            except Exception as e:
                                print(f"Error saving results: {e}")
                            print(
                                f"Results saved to: {episode_steps_filename}"
                            )

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        self.observation_dict = []  # reset
                        self.vla_action = []  # reset
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
