#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import torch
from torch.utils.data import DataLoader

import habitat
from habitat import logger
from habitat_baselines.common.base_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.data import EQADataset
from habitat_baselines.il.metrics import VqaMetric
from habitat_baselines.il.models.models import VqaLstmCnnAttentionModel


@baseline_registry.register_trainer(name="vqa")
class VQATrainer(BaseILTrainer):
    r"""Trainer class for VQA model used in EmbodiedQA (Das et. al.; CVPR 2018)
    Paper: https://embodiedqa.org/paper.pdf.
    """
    supported_tasks = ["VQA-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        if config is not None:
            logger.info(f"config: {config}")

    def train(self) -> None:
        r"""Main method for training VQA (Answering) model of EQA.

        Returns:
            None
        """
        config = self.config

        assert torch.cuda.is_available(), "Cuda-enabled GPU required"
        torch.cuda.set_device(config.TORCH_GPU_ID)

        env = habitat.Env(config=config.TASK_CONFIG)

        vqa_dataset = EQADataset(
            env, config, input_type="vqa", num_frames=config.IL.VQA.num_frames,
        )

        train_loader = DataLoader(
            vqa_dataset, batch_size=config.IL.VQA.batch_size, shuffle=True
        )

        logger.info("train_loader has %d samples" % len(vqa_dataset))

        q_vocab_dict, ans_vocab_dict = vqa_dataset.get_vocab_dicts()

        model_kwargs = {
            "q_vocab": q_vocab_dict,
            "ans_vocab": ans_vocab_dict,
            "edfe_ckpt_path": config.EDFE_CKPT_PATH,
        }
        model = VqaLstmCnnAttentionModel(**model_kwargs)

        lossFn = torch.nn.CrossEntropyLoss()

        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.IL.VQA.lr),
        )

        metrics = VqaMetric(
            info={"split": "train"},
            metric_names=[
                "loss",
                "accuracy",
                "mean_rank",
                "mean_reciprocal_rank",
            ],
            log_json=os.path.join(config.OUTPUT_LOG_DIR, "train.json"),
        )

        t, epoch = 0, 1

        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_mean_rank = 0.0
        avg_mean_reciprocal_rank = 0.0
        print(model)
        model.double().train().cuda()

        if config.IL.VQA.freeze_encoder:
            model.cnn.eval()

        with TensorboardWriter(
            config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while epoch <= int(config.IL.VQA.max_epochs):
                start_time = time.time()
                for batch in train_loader:
                    t += 1

                    idx, questions, answers, frame_queue = batch

                    optim.zero_grad()

                    questions = questions.cuda()
                    answers = answers.cuda()
                    frame_queue = frame_queue.cuda()

                    scores, att_probs = model(frame_queue, questions)
                    loss = lossFn(scores, answers)

                    # update metrics
                    accuracy, ranks = metrics.compute_ranks(
                        scores.data.cpu(), answers
                    )
                    metrics.update([loss.item(), accuracy, ranks, 1.0 / ranks])

                    loss.backward()
                    optim.step()

                    (
                        metrics_loss,
                        accuracy,
                        mean_rank,
                        mean_reciprocal_rank,
                    ) = metrics.get_stats()

                    avg_loss += metrics_loss
                    avg_accuracy += accuracy
                    avg_mean_rank += mean_rank
                    avg_mean_reciprocal_rank += mean_reciprocal_rank

                    if t % config.LOG_INTERVAL == 0:
                        print("Epoch", epoch)
                        print(metrics.get_stat_string())

                        writer.add_scalar("loss", metrics_loss, t)
                        writer.add_scalar("accuracy", accuracy, t)
                        writer.add_scalar("mean_rank", mean_rank, t)
                        writer.add_scalar(
                            "mean_reciprocal_rank", mean_reciprocal_rank, t
                        )

                        metrics.dump_log()

                avg_loss /= len(train_loader)
                avg_accuracy /= len(train_loader)
                avg_mean_rank /= len(train_loader)
                avg_mean_reciprocal_rank /= len(train_loader)

                end_time = time.time()
                time_taken = "%.01f" % ((end_time - start_time) / 60)

                logger.info(
                    "Epoch {} completed. Time taken: {} minutes.".format(
                        epoch, time_taken
                    )
                )

                logger.info("Average loss: %.02f" % avg_loss)
                logger.info("Average accuracy: %.02f" % avg_accuracy)
                logger.info("Average mean rank: %.02f" % avg_mean_rank)
                logger.info(
                    "Average mean reciprocal rank: %.02f"
                    % avg_mean_reciprocal_rank
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

        assert torch.cuda.is_available(), "Cuda-enabled GPU required"
        torch.cuda.set_device(config.TORCH_GPU_ID)

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        config.freeze()

        env = habitat.Env(config=config.TASK_CONFIG)

        vqa_dataset = EQADataset(
            env, config, input_type="vqa", num_frames=config.IL.VQA.num_frames,
        )

        eval_loader = DataLoader(
            vqa_dataset, batch_size=config.IL.VQA.batch_size, shuffle=False
        )

        logger.info("eval_loader has %d samples" % len(vqa_dataset))

        q_vocab_dict, ans_vocab_dict = vqa_dataset.get_vocab_dicts()

        model_kwargs = {
            "q_vocab": q_vocab_dict,
            "ans_vocab": ans_vocab_dict,
            "edfe_ckpt_path": config.EDFE_CKPT_PATH,
        }
        model = VqaLstmCnnAttentionModel(**model_kwargs)

        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

        lossFn = torch.nn.CrossEntropyLoss()

        t = 0

        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_mean_rank = 0.0
        avg_mean_reciprocal_rank = 0.0

        model.eval()
        model.cnn.eval()
        model.double().cuda()

        metrics = VqaMetric(
            info={"split": "val"},
            metric_names=[
                "loss",
                "accuracy",
                "mean_rank",
                "mean_reciprocal_rank",
            ],
            log_json=os.path.join(config.OUTPUT_LOG_DIR, "eval.json"),
        )
        with torch.no_grad():
            for batch in eval_loader:
                t += 1
                idx, questions, answers, frame_queue = batch
                questions = questions.cuda()
                answers = answers.cuda()
                frame_queue = frame_queue.cuda()

                scores, att_probs = model(frame_queue, questions)

                loss = lossFn(scores, answers)

                accuracy, ranks = metrics.compute_ranks(
                    scores.data.cpu(), answers
                )
                metrics.update([loss.item(), accuracy, ranks, 1.0 / ranks])

                (
                    metrics_loss,
                    accuracy,
                    mean_rank,
                    mean_reciprocal_rank,
                ) = metrics.get_stats(mode=0)

                avg_loss += metrics_loss
                avg_accuracy += accuracy
                avg_mean_rank += mean_rank
                avg_mean_reciprocal_rank += mean_reciprocal_rank

                if t % config.LOG_INTERVAL == 0:
                    print(metrics.get_stat_string(mode=0))
                    metrics.dump_log()

                if config.EVAL_SAVE_RESULTS:
                    if t % config.EVAL_SAVE_RESULTS_INTERVAL == 0:

                        self._save_vqa_results(
                            checkpoint_index,
                            idx,
                            questions,
                            frame_queue,
                            scores,
                            answers,
                            q_vocab_dict,
                            ans_vocab_dict,
                        )

        avg_loss /= len(eval_loader)
        avg_accuracy /= len(eval_loader)
        avg_mean_rank /= len(eval_loader)
        avg_mean_reciprocal_rank /= len(eval_loader)

        writer.add_scalar("avg val loss", avg_loss, checkpoint_index)
        writer.add_scalar("avg val accuracy", avg_accuracy, checkpoint_index)
        writer.add_scalar("avg val mean rank", avg_mean_rank, checkpoint_index)
        writer.add_scalar(
            "avg val mean reciprocal rank",
            avg_mean_reciprocal_rank,
            checkpoint_index,
        )

        logger.info("Average loss: %.02f" % avg_loss)
        logger.info("Average accuracy: %.02f" % avg_accuracy)
        logger.info("Average mean rank: %.02f" % avg_mean_rank)
        logger.info(
            "Average mean reciprocal rank: %.02f" % avg_mean_reciprocal_rank
        )
