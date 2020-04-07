#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import habitat
from habitat import logger
from habitat_baselines.common.base_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.data import EQADataset
from habitat_baselines.il.models.models import (
    NavPlannerControllerModel,
    MaskedNLLCriterion,
)
from habitat_baselines.il.metrics import NavMetric

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@baseline_registry.register_trainer(name="nav")
class NavTrainer(BaseILTrainer):
    r"""Trainer class for Nav model used in EmbodiedQA (Das et. al.;CVPR 2018)
    Paper: https://embodiedqa.org/paper.pdf.
    """
    supported_tasks = ["EQA-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        # if config is not None:
        # logger.info(f"config: {config}")

    def train(self) -> None:
        r"""Main method for training Navigation model of EQA.

        Returns:
            None
        """
        config = self.config

        assert torch.cuda.is_available(), "Cuda-enabled GPU required"
        torch.cuda.set_device(config.TORCH_GPU_ID)

        env = habitat.Env(config=config.TASK_CONFIG)

        nav_dataset = EQADataset(
            env,
            config,
            input_type="pacman",
            max_controller_actions=config.IL.NAV.max_controller_actions,
        )

        train_loader = DataLoader(
            nav_dataset, batch_size=config.IL.NAV.batch_size, shuffle=True
        )

        logger.info("train_loader has %d samples" % len(nav_dataset))

        q_vocab_dict, _ = nav_dataset.get_vocab_dicts()

        model_kwargs = {"q_vocab": q_vocab_dict}
        model = NavPlannerControllerModel(**model_kwargs)

        planner_lossFn = MaskedNLLCriterion()
        controller_lossFn = MaskedNLLCriterion()

        metrics = NavMetric(
            info={"split": "train"},
            metric_names=["planner_loss", "controller_loss"],
            log_json=os.path.join(config.OUTPUT_LOG_DIR, "train.json"),
        )

        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.IL.NAV.lr),
        )

        t, epoch = 0, 1

        avg_p_loss = 0.0
        avg_c_loss = 0.0

        model.train().cuda()

        with TensorboardWriter(
            config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while epoch <= int(config.IL.NAV.max_epochs):
                start_time = time.time()
                for batch in train_loader:
                    t += 1

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

                    questions = questions.cuda()

                    planner_img_feats = planner_img_feats.cuda()
                    planner_actions_in = planner_actions_in.cuda()
                    planner_actions_out = planner_actions_out.cuda()
                    planner_action_lengths = planner_action_lengths.cuda()
                    planner_masks = planner_masks.cuda()

                    controller_img_feats = controller_img_feats.cuda()
                    controller_actions_in = controller_actions_in.cuda()
                    planner_hidden_idx = planner_hidden_idx.cuda()
                    controller_outs = controller_outs.cuda()
                    controller_action_lengths = (
                        controller_action_lengths.cuda()
                    )
                    controller_masks = controller_masks.cuda()

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

                    planner_scores, controller_scores, planner_hidden = model(
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

                    planner_loss = planner_lossFn(
                        planner_logprob,
                        planner_actions_out[:, : planner_action_lengths.max()]
                        .contiguous()
                        .view(-1, 1),
                        planner_masks[:, : planner_action_lengths.max()]
                        .contiguous()
                        .view(-1, 1),
                    )

                    controller_loss = controller_lossFn(
                        controller_logprob,
                        controller_outs[:, : controller_action_lengths.max()]
                        .contiguous()
                        .view(-1, 1),
                        controller_masks[:, : controller_action_lengths.max()]
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
                        print("Epoch", epoch)
                        print(metrics.get_stat_string())

                        writer.add_scalar("planner loss", planner_loss, t)
                        writer.add_scalar(
                            "controller loss", controller_loss, t
                        )

                        metrics.dump_log()

                avg_p_loss /= len(train_loader)
                avg_c_loss /= len(train_loader)

                end_time = time.time()
                time_taken = "%.01f" % ((end_time - start_time) / 60)

                logger.info(
                    "Epoch {} completed. Time taken: {} minutes.".format(
                        epoch, time_taken
                    )
                )

                logger.info("Average planner loss: %.02f" % avg_p_loss)
                logger.info("Average controller loss: %.02f" % avg_c_loss)

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

        assert torch.cuda.is_available(), "Cuda-enabled GPU required"
        torch.cuda.set_device(config.TORCH_GPU_ID)

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        config.freeze()

        env = habitat.Env(config=config.TASK_CONFIG)

        nav_dataset = EQADataset(
            env,
            config,
            input_type="pacman",
            max_controller_actions=config.IL.NAV.max_controller_actions,
        )

        eval_loader = DataLoader(
            nav_dataset,
            batch_size=config.IL.NAV.eval_batch_size,
            shuffle=False,
        )

        logger.info("eval_loader has %d samples" % len(nav_dataset))

        q_vocab_dict, ans_vocab_dict = nav_dataset.get_vocab_dicts()

        model_kwargs = {"q_vocab": q_vocab_dict}
        model = NavPlannerControllerModel(**model_kwargs)

        invalids = []

        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        model.eval().cuda()

        results_dir = config.RESULTS_DIR.format(split="val")

        metrics = NavMetric(
            info={"split": "val"},
            metric_names=[
                "d_0_10",
                "d_0_30",
                "d_0_50",
                "d_T_10",
                "d_T_30",
                "d_T_50",
                "d_D_10",
                "d_D_30",
                "d_D_50",
                "d_min_10",
                "d_min_30",
                "d_min_50",
                "stop_10",
                "stop_30",
                "stop_50",
                "ep_len_10",
                "ep_len_30",
                "ep_len_50",
            ],
            log_json=os.path.join(config.OUTPUT_LOG_DIR, "eval.json"),
        )

        t = 0

        for batch in eval_loader:
            t += 1

            idx, question, answer, actions, action_length, goal_pos = batch

            metrics_slug = {}

            for i in [10, 30, 50]:

                if i > action_length.item():
                    invalids.append([idx.item(), i])
                    continue

                question = question.cuda()

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
                    actions[0, : action_length[0] + 1].numpy(),
                    i,
                    config.IL.NAV.max_controller_actions,
                )

                planner_actions_in = planner_actions_in.cuda()
                planner_img_feats = planner_img_feats.cuda()

                # forward planner till spawn to update hidden state
                for step in range(planner_actions_in.size(0)):

                    planner_scores, planner_hidden = model.planner_step(
                        question,
                        planner_img_feats[step].unsqueeze(0).unsqueeze(0),
                        planner_actions_in[step].view(1, 1),
                        planner_hidden,
                    )

                env.sim.set_agent_state(init_pos.position, init_pos.rotation)
                init_dist_to_target = env.sim.geodesic_distance(
                    init_pos.position, goal_pos
                )

                if init_dist_to_target < 0 or init_dist_to_target == float(
                    "inf"
                ):  # unreachable
                    invalids.append([idx.item(), i])
                    continue

                dists_to_target, pos_queue = [init_dist_to_target], [init_pos]

                planner_actions, controller_actions = [], []

                episode_length = 0
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
                imgs = []

                for step in range(config.IL.NAV.max_episode_length):
                    if not first_step:
                        imgs.append(img)
                        img = (
                            torch.from_numpy(img.transpose(2, 0, 1)).float()
                            / 255.0
                        )
                        img_feat = eval_loader.dataset.cnn(
                            img.view(1, 3, 256, 256).cuda()
                        ).view(1, 1, 4608)
                    else:
                        img_feat = controller_img_feats.cuda().view(1, 1, 4608)

                    if not first_step or first_step_is_controller:
                        # query controller to continue or not
                        controller_action_in = (
                            torch.LongTensor(1, 1).fill_(action).cuda()
                        )
                        controller_scores = model.controller_step(
                            img_feat, controller_action_in, planner_hidden[0],
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
                                torch.LongTensor(1, 1).fill_(action + 1).cuda()
                            )
                            (
                                planner_scores,
                                planner_hidden,
                            ) = model.planner_step(
                                question, img_feat, action_in, planner_hidden,
                            )
                        prob = F.softmax(planner_scores, dim=1)
                        action = int(prob.max(1)[1].data.cpu().numpy()[0])
                        planner_actions.append(action)

                    episode_done = (
                        action == 3
                        or episode_length >= config.IL.NAV.max_episode_length
                    )

                    episode_length += 1
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
                print("Number of invalids: ", len(invalids) / 3)

                logger.info(
                    "EVAL: Avg metrics: {}".format(
                        metrics.get_stat_string(mode=0)
                    )
                )
                print("-----------------------------------------------------")

            if config.EVAL_SAVE_RESULTS:
                if t % config.EVAL_SAVE_RESULTS_INTERVAL == 0:
                    q_string = ""
                    for token in question[0]:
                        if token != 0:
                            for word, indx in q_vocab_dict.items():
                                if indx == token:
                                    q_word = word
                                    break
                            q_string += q_word + " "

                        else:
                            break
                    q_string += "?"
                    print("Question:", q_string)

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
