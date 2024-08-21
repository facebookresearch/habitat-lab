#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from datetime import datetime
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import habitat
from habitat import logger
from habitat.datasets.utils import VocabDict
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.nav_data import NavDataset
from habitat_baselines.il.metrics import NavMetric
from habitat_baselines.il.models.models import (
    MaskedNLLCriterion,
    NavPlannerControllerModel,
)
from habitat_baselines.utils.common import generate_video


@baseline_registry.register_trainer(name="pacman")
class PACMANTrainer(BaseILTrainer):
    r"""Trainer class for PACMAN (Planner and Controller Module) Nav model
    used in EmbodiedQA (Das et. al.;CVPR 2018)
    Paper: https://embodiedqa.org/paper.pdf.
    """
    supported_tasks = ["EQA-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if config is not None:
            logger.info(f"config: {config}")

    def _save_nav_results(
        self,
        ckpt_path: str,
        ep_id: int,
        questions: torch.Tensor,
        imgs: List[np.ndarray],
        q_vocab_dict: VocabDict,
        results_dir: str,
        writer: TensorboardWriter,
        video_option: list,
    ) -> None:
        r"""For saving NAV-PACMAN eval results.
        Args:
            ckpt_path: path of checkpoint being evaluated
            ep_id: episode id (batch index)
            questions: input question to model
            imgs: images' tensor containing input frames
            q_vocab_dict: question vocab dictionary
            results_dir: dir to save results
            writer: tensorboard writer
            video_option: ["disk", "tb"]
        Returns:
            None
        """

        question = questions[0]

        ckpt_epoch = ckpt_path[ckpt_path.rfind("/") + 1 : -5]
        results_dir = os.path.join(results_dir, ckpt_epoch)
        ckpt_no = int(ckpt_epoch[6:])

        q_string = q_vocab_dict.token_idx_2_string(question)
        frames_with_text: List[np.ndarray] = []
        for frame in imgs:
            border_width = 32
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 0, 0)
            scale = 0.3
            thickness = 1

            frame = cv2.copyMakeBorder(
                frame,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

            frame = cv2.putText(
                frame,
                "Question: " + q_string,
                (10, 15),
                font,
                scale,
                color,
                thickness,
            )

            frames_with_text.append(np.ndarray(frame))
        generate_video(
            video_option,
            results_dir,
            frames_with_text,
            ep_id,
            ckpt_no,
            {},
            writer,
            fps=5,
        )

    def train(self) -> None:
        r"""Main method for training Navigation model of EQA.

        Returns:
            None
        """
        config = self.config

        with habitat.Env(config.habitat) as env:
            nav_dataset = (
                NavDataset(
                    config,
                    env,
                    self.device,
                )
                .shuffle(1000)
                .decode("rgb")
            )

            nav_dataset = nav_dataset.map(nav_dataset.map_dataset_sample)

            train_loader = DataLoader(
                nav_dataset,
                batch_size=config.habitat_baselines.il.nav.batch_size,
            )

            logger.info("train_loader has {} samples".format(len(nav_dataset)))

            q_vocab_dict, _ = nav_dataset.get_vocab_dicts()

            model_kwargs = {"q_vocab": q_vocab_dict.word2idx_dict}
            model = NavPlannerControllerModel(**model_kwargs)

            planner_loss_fn = MaskedNLLCriterion()
            controller_loss_fn = MaskedNLLCriterion()

            optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(config.habitat_baselines.il.nav.lr),
            )

            metrics = NavMetric(
                info={"split": "train"},
                metric_names=["planner_loss", "controller_loss"],
                log_json=os.path.join(
                    config.habitat_baselines.il.output_log_dir, "train.json"
                ),
            )

            epoch = 1

            avg_p_loss = 0.0
            avg_c_loss = 0.0

            logger.info(model)
            model.train().to(self.device)

            with TensorboardWriter(
                "train_{}/{}".format(
                    config.habitat_baselines.tensorboard_dir,
                    datetime.today().strftime("%Y-%m-%d-%H:%M"),
                ),
                flush_secs=self.flush_secs,
            ) as writer:
                while epoch <= config.habitat_baselines.il.nav.max_epochs:
                    start_time = time.time()
                    for t, batch in enumerate(train_loader):
                        batch = (
                            item.to(self.device, non_blocking=True)
                            for item in batch
                        )
                        (
                            idx,
                            questions,
                            _,
                            planner_img_feats,
                            planner_actions_in,
                            planner_actions_out,
                            planner_action_lengths,
                            planner_masks,
                            controller_img_feats,
                            controller_actions_in,
                            planner_hidden_idx,
                            controller_outs,
                            controller_action_lengths,
                            controller_masks,
                        ) = batch

                        (
                            planner_action_lengths,
                            perm_idx,
                        ) = planner_action_lengths.sort(0, descending=True)
                        questions = questions[perm_idx]

                        planner_img_feats = planner_img_feats[perm_idx]
                        planner_actions_in = planner_actions_in[perm_idx]
                        planner_actions_out = planner_actions_out[perm_idx]
                        planner_masks = planner_masks[perm_idx]

                        controller_img_feats = controller_img_feats[perm_idx]
                        controller_actions_in = controller_actions_in[perm_idx]
                        controller_outs = controller_outs[perm_idx]
                        planner_hidden_idx = planner_hidden_idx[perm_idx]
                        controller_action_lengths = controller_action_lengths[
                            perm_idx
                        ]
                        controller_masks = controller_masks[perm_idx]

                        (
                            planner_scores,
                            controller_scores,
                            planner_hidden,
                        ) = model(
                            questions,
                            planner_img_feats,
                            planner_actions_in,
                            planner_action_lengths.cpu().numpy(),
                            planner_hidden_idx,
                            controller_img_feats,
                            controller_actions_in,
                            controller_action_lengths,
                        )

                        planner_logprob = F.log_softmax(planner_scores, dim=1)
                        controller_logprob = F.log_softmax(
                            controller_scores, dim=1
                        )

                        planner_loss = planner_loss_fn(
                            planner_logprob,
                            planner_actions_out[
                                :, : planner_action_lengths.max()
                            ].reshape(-1, 1),
                            planner_masks[
                                :, : planner_action_lengths.max()
                            ].reshape(-1, 1),
                        )

                        controller_loss = controller_loss_fn(
                            controller_logprob,
                            controller_outs[
                                :, : controller_action_lengths.max()
                            ].reshape(-1, 1),
                            controller_masks[
                                :, : controller_action_lengths.max()
                            ].reshape(-1, 1),
                        )

                        # zero grad
                        optim.zero_grad()

                        # update metrics
                        metrics.update(
                            [planner_loss.item(), controller_loss.item()]
                        )

                        (planner_loss + controller_loss).backward()

                        optim.step()

                        (planner_loss, controller_loss) = metrics.get_stats()

                        avg_p_loss += planner_loss
                        avg_c_loss += controller_loss

                        if t % config.habitat_baselines.log_interval == 0:
                            logger.info("Epoch: {}".format(epoch))
                            logger.info(metrics.get_stat_string())

                            writer.add_scalar("planner loss", planner_loss, t)
                            writer.add_scalar(
                                "controller loss", controller_loss, t
                            )

                            metrics.dump_log()

                    # Dataloader length for IterableDataset doesn't take into
                    # account batch size for Pytorch v < 1.6.0
                    num_batches = math.ceil(
                        len(nav_dataset)
                        / config.habitat_baselines.il.nav.batch_size
                    )

                    avg_p_loss /= num_batches
                    avg_c_loss /= num_batches

                    end_time = time.time()
                    time_taken = "{:.1f}".format((end_time - start_time) / 60)

                    logger.info(
                        "Epoch {} completed. Time taken: {} minutes.".format(
                            epoch, time_taken
                        )
                    )

                    logger.info(
                        "Average planner loss: {:.2f}".format(avg_p_loss)
                    )
                    logger.info(
                        "Average controller loss: {:.2f}".format(avg_c_loss)
                    )

                    print("-----------------------------------------")

                    if (
                        epoch % config.habitat_baselines.checkpoint_interval
                        == 0
                    ):
                        self.save_checkpoint(
                            model.state_dict(), "epoch_{}.ckpt".format(epoch)
                        )

                    epoch += 1

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        config = self.config

        with habitat.config.read_write(config):
            config.habitat.dataset.split = (
                self.config.habitat_baselines.eval.split
            )

        with habitat.Env(config.habitat) as env:
            nav_dataset = NavDataset(
                config,
                env,
                self.device,
            ).decode("rgb")

            nav_dataset = nav_dataset.map(nav_dataset.map_dataset_sample)

            eval_loader = DataLoader(nav_dataset)

            logger.info("eval_loader has {} samples".format(len(nav_dataset)))

            q_vocab_dict, ans_vocab_dict = nav_dataset.get_vocab_dicts()

            model_kwargs = {"q_vocab": q_vocab_dict.word2idx_dict}
            model = NavPlannerControllerModel(**model_kwargs)

            invalids = []

            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
            model.eval().to(self.device)

            results_dir = config.habitat_baselines.il.results_dir.format(
                split="val"
            )
            video_option = self.config.habitat_baselines.eval.video_option

            metrics = NavMetric(
                info={"split": "val"},
                metric_names=[
                    "{}_{}".format(y, x)
                    for x in [10, 30, 50, "rand_init"]
                    for z in ["", "_f"]
                    for y in [
                        *["d_{}{}".format(k, z) for k in [0, "T", "D", "min"]],
                        *[w for w in ["stop", "ep_len"] if z == ""],
                    ]
                ],
                log_json=os.path.join(
                    config.habitat_baselines.il.output_log_dir, "eval.json"
                ),
            )

            for t, batch in enumerate(eval_loader):
                idx, question, answer, actions, action_length, goal_pos = batch

                metrics_slug = {}
                imgs = []  # type:ignore
                for i in [10, 30, 50, "rand_init"]:
                    for j in ["pred", "fwd-only"]:
                        question = question.to(self.device)

                        controller_step = False
                        planner_hidden = model.planner_nav_rnn.init_hidden(1)

                        # get hierarchical action history
                        (
                            planner_actions_in,
                            planner_img_feats,
                            controller_step,
                            controller_action_in,
                            controller_img_feats,
                            init_pos,
                            controller_action_counter,
                        ) = nav_dataset.get_hierarchical_features_till_spawn(
                            idx.item(),
                            actions[0, : action_length.item()].numpy(),
                            i if i != "rand_init" else action_length.item(),
                            config.habitat_baselines.il.nav.max_controller_actions,
                        )
                        if j == "pred":
                            planner_actions_in = planner_actions_in.to(
                                self.device
                            )
                            planner_img_feats = planner_img_feats.to(
                                self.device
                            )

                            for step in range(planner_actions_in.size(0)):
                                (
                                    planner_scores,
                                    planner_hidden,
                                ) = model.planner_step(
                                    question,
                                    planner_img_feats[step][
                                        (None,) * 2
                                    ],  # unsqueezing twice
                                    planner_actions_in[step].view(1, 1),
                                    planner_hidden,
                                )

                        env.sim.set_agent_state(
                            init_pos.position, init_pos.rotation
                        )
                        init_dist_to_target = env.sim.geodesic_distance(
                            init_pos.position, goal_pos
                        )

                        if (
                            init_dist_to_target < 0
                            or init_dist_to_target == float("inf")
                        ):  # unreachable
                            invalids.append([idx.item(), i])
                            continue

                        dists_to_target, pos_queue = [init_dist_to_target], [
                            init_pos
                        ]
                        if j == "pred":
                            planner_actions, controller_actions = [], []

                            if (
                                config.habitat_baselines.il.nav.max_controller_actions
                                > 1
                            ):
                                controller_action_counter = (
                                    controller_action_counter
                                    % config.habitat_baselines.il.nav.max_controller_actions
                                )
                                controller_action_counter = max(
                                    controller_action_counter - 1, 0
                                )
                            else:
                                controller_action_counter = 0

                            first_step = True
                            first_step_is_controller = controller_step
                            planner_step = True
                            action = int(controller_action_in)

                        img = None
                        for episode_length in range(
                            config.habitat_baselines.il.nav.max_episode_length
                        ):
                            if j == "pred":
                                if not first_step:
                                    if (
                                        i == 30
                                    ):  # saving results for 30-step walked back case
                                        imgs.append(img)
                                    img_feat = (
                                        eval_loader.dataset.get_img_features(
                                            img, preprocess=True
                                        ).view(1, 1, 4608)
                                    )
                                else:
                                    img_feat = controller_img_feats.to(
                                        self.device
                                    ).view(1, 1, 4608)

                                if not first_step or first_step_is_controller:
                                    # query controller to continue or not
                                    controller_action_in = (
                                        torch.LongTensor(1, 1)
                                        .fill_(action)
                                        .to(self.device)
                                    )
                                    controller_scores = model.controller_step(
                                        img_feat,
                                        controller_action_in,
                                        planner_hidden[0],
                                    )

                                    prob = F.softmax(controller_scores, dim=1)
                                    controller_action = int(
                                        prob.max(1)[1].data.cpu().numpy()[0]
                                    )

                                    if (
                                        controller_action == 1
                                        and controller_action_counter
                                        < config.habitat_baselines.il.nav.max_controller_actions
                                        - 1
                                    ):
                                        controller_action_counter += 1
                                        planner_step = False
                                    else:
                                        controller_action_counter = 0
                                        planner_step = True
                                        controller_action = 0

                                    controller_actions.append(
                                        controller_action
                                    )
                                    first_step = False

                                if planner_step:
                                    if not first_step:
                                        action_in = (
                                            torch.LongTensor(1, 1)
                                            .fill_(action + 1)
                                            .to(self.device)
                                        )
                                        (
                                            planner_scores,
                                            planner_hidden,
                                        ) = model.planner_step(
                                            question,
                                            img_feat,
                                            action_in,
                                            planner_hidden,
                                        )
                                    prob = F.softmax(planner_scores, dim=1)
                                    action = int(
                                        prob.max(1)[1].data.cpu().numpy()[0]
                                    )
                                    planner_actions.append(action)

                            else:
                                action = 0

                            episode_done = (
                                action == 3
                                or episode_length
                                >= config.habitat_baselines.il.nav.max_episode_length
                            )

                            agent_pos = env.sim.get_agent_state().position

                            dists_to_target.append(
                                env.sim.geodesic_distance(agent_pos, goal_pos)
                            )
                            pos_queue.append([agent_pos])

                            if episode_done:
                                break

                            if action == 0:
                                my_action = 1  # forward
                            elif action == 1:
                                my_action = 2  # left
                            elif action == 2:
                                my_action = 3  # right
                            elif action == 3:
                                my_action = 0  # stop

                            observations = env.sim.step(my_action)
                            img = observations["rgb"]
                            first_step = False

                        # compute stats
                        m = "" if j == "pred" else "_f"
                        metrics_slug[
                            "d_T{}_{}".format(m, i)
                        ] = dists_to_target[-1]
                        metrics_slug["d_D{}_{}".format(m, i)] = (
                            dists_to_target[0] - dists_to_target[-1]
                        )
                        metrics_slug["d_min{}_{}".format(m, i)] = np.array(
                            dists_to_target
                        ).min()

                        if j != "fwd-only":
                            metrics_slug[
                                "ep_len_{}".format(i)
                            ] = episode_length
                            if action == 3:
                                metrics_slug["stop_{}".format(i)] = 1
                            else:
                                metrics_slug["stop_{}".format(i)] = 0

                            metrics_slug["d_0_{}".format(i)] = dists_to_target[
                                0
                            ]

                # collate and update metrics
                metrics_list = []
                for ind, i in enumerate(metrics.metric_names):
                    if i not in metrics_slug:
                        metrics_list.append(metrics.metrics[ind][0])
                    else:
                        metrics_list.append(metrics_slug[i])

                # update metrics
                metrics.update(metrics_list)

                if t % config.habitat_baselines.log_interval == 0:
                    logger.info(
                        "Valid cases: {}; Invalid cases: {}".format(
                            (t + 1) * 8 - len(invalids), len(invalids)
                        )
                    )
                    logger.info(
                        "eval: Avg metrics: {}".format(
                            metrics.get_stat_string(mode=0)
                        )
                    )
                    print(
                        "-----------------------------------------------------"
                    )

                if (
                    config.habitat_baselines.il.eval_save_results
                    and t
                    % config.habitat_baselines.il.eval_save_results_interval
                    == 0
                ):
                    q_string = q_vocab_dict.token_idx_2_string(question[0])
                    logger.info("Question: {}".format(q_string))

                    self._save_nav_results(
                        checkpoint_path,
                        t,
                        question,
                        imgs,
                        q_vocab_dict,
                        results_dir,
                        writer,
                        video_option,
                    )
