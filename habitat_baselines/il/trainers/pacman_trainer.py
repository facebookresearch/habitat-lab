#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import habitat
from habitat import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.nav_data import NavDataset
from habitat_baselines.il.metrics import NavMetric
from habitat_baselines.il.models.models import (
    MaskedNLLCriterion,
    NavPlannerControllerModel,
)

cv2 = try_cv2_import()


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
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if config is not None:
            logger.info(f"config: {config}")

    def _save_nav_results(
        self,
        ckpt_path: int,
        t: int,
        questions: torch.Tensor,
        imgs: List[np.ndarray],
        q_vocab_dict: Dict,
        results_dir: str,
    ) -> None:

        r"""For saving VQA results.
        Args:
            ckpt_path: path of checkpoint being evaluated
            t: index
            images: images' tensor containing input frames
            question: input question to model
        Returns:
            None
        """

        question = questions[0]

        ckpt_epoch = ckpt_path[ckpt_path.rfind("/") + 1 :]
        results_dir = os.path.join(results_dir, ckpt_epoch)

        q_string = q_vocab_dict.token_idx_2_string(question)
        frames_with_text = []
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

            frames_with_text.append(frame)

        images_to_video(frames_with_text, results_dir, "ep_" + str(t))

    def train(self) -> None:
        r"""Main method for training Navigation model of EQA.

        Returns:
            None
        """
        config = self.config

        with habitat.Env(config.TASK_CONFIG) as env:
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
                nav_dataset, batch_size=config.IL.NAV.batch_size
            )

            logger.info("train_loader has {} samples".format(len(nav_dataset)))

            q_vocab_dict, _ = nav_dataset.get_vocab_dicts()

            model_kwargs = {"q_vocab": q_vocab_dict.word2idx_dict}
            model = NavPlannerControllerModel(**model_kwargs)

            planner_loss_fn = MaskedNLLCriterion()
            controller_loss_fn = MaskedNLLCriterion()

            optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(config.IL.NAV.lr),
            )

            metrics = NavMetric(
                info={"split": "train"},
                metric_names=["planner_loss", "controller_loss"],
                log_json=os.path.join(config.OUTPUT_LOG_DIR, "train.json"),
            )

            epoch = 1

            avg_p_loss = 0.0
            avg_c_loss = 0.0

            logger.info(model)
            model.train().to(self.device)

            with TensorboardWriter(
                config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) as writer:
                while epoch <= config.IL.NAV.max_epochs:
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
                            ]
                            .contiguous()
                            .view(-1, 1),
                            planner_masks[:, : planner_action_lengths.max()]
                            .contiguous()
                            .view(-1, 1),
                        )

                        controller_loss = controller_loss_fn(
                            controller_logprob,
                            controller_outs[
                                :, : controller_action_lengths.max()
                            ]
                            .contiguous()
                            .view(-1, 1),
                            controller_masks[
                                :, : controller_action_lengths.max()
                            ]
                            .contiguous()
                            .view(-1, 1),
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

                        if t % config.LOG_INTERVAL == 0:
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
                        len(nav_dataset) / config.IL.NAV.batch_size
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

                    if epoch % config.CHECKPOINT_INTERVAL == 0:
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

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        config.freeze()

        with habitat.Env(config.TASK_CONFIG) as env:
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

            results_dir = config.RESULTS_DIR.format(split="val")

            metrics = NavMetric(
                info={"split": "val"},
                metric_names=[
                    "{}_{}".format(y, x)
                    for x in [10, 30, 50]
                    for y in [
                        *["d_{}".format(x) for x in [0, "T", "D", "min"]],
                        "stop",
                        "ep_len",
                    ]
                ],
                log_json=os.path.join(config.OUTPUT_LOG_DIR, "eval.json"),
            )

            for t, batch in enumerate(eval_loader):
                idx, question, answer, actions, action_length, goal_pos = batch

                metrics_slug = {}

                for i in [10, 30, 50]:
                    if i > action_length.item():
                        invalids.append([idx.item(), i])
                        continue

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
                        actions[0, : action_length[0]].numpy(),
                        i,
                        config.IL.NAV.max_controller_actions,
                    )

                    planner_actions_in = planner_actions_in.to(self.device)
                    planner_img_feats = planner_img_feats.to(self.device)

                    # forward planner till spawn to update hidden state
                    for step in range(planner_actions_in.size(0)):

                        planner_scores, planner_hidden = model.planner_step(
                            question,
                            planner_img_feats[step].unsqueeze(0).unsqueeze(0),
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

                    planner_actions, controller_actions = [], []

                    if config.IL.NAV.max_controller_actions > 1:
                        controller_action_counter = (
                            controller_action_counter
                            % config.IL.NAV.max_controller_actions
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

                    imgs = []
                    for episode_length in range(
                        config.IL.NAV.max_episode_length
                    ):
                        if not first_step:
                            imgs.append(img)
                            img = (
                                torch.from_numpy(
                                    img.transpose(2, 0, 1)
                                ).float()
                                / 255.0
                            )
                            img_feat = eval_loader.dataset.cnn(
                                img.view(1, 3, 256, 256).to(self.device)
                            ).view(1, 1, 4608)
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
                                < config.IL.NAV.max_controller_actions - 1
                            ):
                                controller_action_counter += 1
                                planner_step = False
                            else:
                                controller_action_counter = 0
                                planner_step = True
                                controller_action = 0

                            controller_actions.append(controller_action)
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
                            action = int(prob.max(1)[1].data.cpu().numpy()[0])
                            planner_actions.append(action)

                        episode_done = (
                            action == 3
                            or episode_length
                            >= config.IL.NAV.max_episode_length
                        )

                        dists_to_target.append(
                            env.sim.geodesic_distance(
                                env.sim.get_agent_state().position, goal_pos
                            )
                        )

                        pos_queue.append([env.sim.get_agent_state()])

                        if episode_done:
                            break

                        if action == 0:
                            my_action = 1
                        elif action == 1:
                            my_action = 2
                        elif action == 2:
                            my_action = 3
                        elif action == 3:
                            my_action = 0

                        observations = env.sim.step(my_action)
                        img = observations["rgb"]
                        first_step = False

                    # compute stats
                    metrics_slug["d_0_" + str(i)] = dists_to_target[0]
                    metrics_slug["d_T_" + str(i)] = dists_to_target[-1]
                    metrics_slug["d_D_" + str(i)] = (
                        dists_to_target[0] - dists_to_target[-1]
                    )
                    metrics_slug["d_min_" + str(i)] = np.array(
                        dists_to_target
                    ).min()
                    metrics_slug["ep_len_" + str(i)] = episode_length
                    if action == 3:
                        metrics_slug["stop_" + str(i)] = 1
                    else:
                        metrics_slug["stop_" + str(i)] = 0

                # collate and update metrics
                metrics_list = []
                for i in metrics.metric_names:
                    if i not in metrics_slug:
                        metrics_list.append(
                            metrics.metrics[metrics.metric_names.index(i)][0]
                        )
                    else:
                        metrics_list.append(metrics_slug[i])

                # update metrics
                metrics.update(metrics_list)

                if t % config.LOG_INTERVAL == 0:
                    logger.info(
                        "Number of invalid cases: {}".format(len(invalids))
                    )
                    logger.info(
                        "Number of valid cases: {}".format(
                            (t + 1) * 3 - len(invalids)
                        )
                    )

                    logger.info(
                        "EVAL: Avg metrics: {}".format(
                            metrics.get_stat_string(mode=0)
                        )
                    )
                    print(
                        "-----------------------------------------------------"
                    )

                if (
                    config.EVAL_SAVE_RESULTS
                    and t % config.EVAL_SAVE_RESULTS_INTERVAL == 0
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
                    )

            eval_accuracy = metrics.metrics[8][0]  # d_D_50
            logger.info(
                "EVAL: [Eval accuracy (d_D_50): {:.04f}]".format(eval_accuracy)
            )
