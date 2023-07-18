#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from habitat import logger
from habitat.config import read_write
from habitat.datasets.utils import VocabDict
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.data import EQADataset
from habitat_baselines.il.metrics import VqaMetric
from habitat_baselines.il.models.models import VqaLstmCnnAttentionModel
from habitat_baselines.utils.common import img_bytes_2_np_array
from habitat_baselines.utils.visualizations.utils import save_vqa_image_results


@baseline_registry.register_trainer(name="vqa")
class VQATrainer(BaseILTrainer):
    r"""Trainer class for VQA model used in EmbodiedQA (Das et. al.; CVPR 2018)
    Paper: https://embodiedqa.org/paper.pdf.
    """
    supported_tasks = ["VQA-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if config is not None:
            logger.info(f"config: {OmegaConf.to_yaml(config)}")

    def _make_results_dir(self):
        r"""Makes directory for saving VQA eval results."""
        dir_name = self.config.habitat_baselines.il.results_dir.format(
            split="val"
        )
        os.makedirs(dir_name, exist_ok=True)

    def _save_vqa_results(
        self,
        ckpt_idx: int,
        episode_ids: torch.Tensor,
        questions: torch.Tensor,
        images: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_answers: torch.Tensor,
        q_vocab_dict: VocabDict,
        ans_vocab_dict: VocabDict,
    ) -> None:
        r"""For saving VQA results.
        Args:
            ckpt_idx: idx of checkpoint being evaluated
            episode_ids: episode ids of batch
            questions: input questions to model
            images: images' tensor containing input frames
            pred_scores: model prediction scores
            gt_answers: ground truth answers
            ground_truth: ground truth answer
            q_vocab_dict: Question VocabDict
            ans_vocab_dict: Answer VocabDict

        Returns:
            None
        """
        episode_id = episode_ids[0].item()
        question = questions[0]
        images = images[0]
        gt_answer = gt_answers[0]
        scores = pred_scores[0]

        q_string = q_vocab_dict.token_idx_2_string(question)

        _, index = scores.max(0)
        pred_answer = sorted(ans_vocab_dict.word2idx_dict.keys())[index]
        gt_answer = sorted(ans_vocab_dict.word2idx_dict.keys())[gt_answer]

        logger.info("Question: {}".format(q_string))
        logger.info("Predicted answer: {}".format(pred_answer))
        logger.info("Ground-truth answer: {}".format(gt_answer))

        result_path = self.config.habitat_baselines.il.results_dir.format(
            split=self.config.habitat.dataset.split
        )

        result_path = os.path.join(
            result_path, "ckpt_{}_{}_image.jpg".format(ckpt_idx, episode_id)
        )

        save_vqa_image_results(
            images, q_string, pred_answer, gt_answer, result_path
        )

    def train(self) -> None:
        r"""Main method for training VQA (Answering) model of EQA.

        Returns:
            None
        """
        config = self.config

        # env = habitat.Env(config=config.habitat)

        vqa_dataset = (
            EQADataset(
                config,
                input_type="vqa",
                num_frames=config.habitat_baselines.il.vqa.num_frames,
            )
            .shuffle(1000)
            .to_tuple(
                "episode_id",
                "question",
                "answer",
                *["{0:0=3d}.jpg".format(x) for x in range(0, 5)],
            )
            .map(img_bytes_2_np_array)
        )

        train_loader = DataLoader(
            vqa_dataset, batch_size=config.habitat_baselines.il.vqa.batch_size
        )

        logger.info("train_loader has {} samples".format(len(vqa_dataset)))

        q_vocab_dict, ans_vocab_dict = vqa_dataset.get_vocab_dicts()

        model_kwargs = {
            "q_vocab": q_vocab_dict.word2idx_dict,
            "ans_vocab": ans_vocab_dict.word2idx_dict,
            "eqa_cnn_pretrain_ckpt_path": config.habitat_baselines.il.eqa_cnn_pretrain_ckpt_path,
            "freeze_encoder": config.habitat_baselines.il.vqa.freeze_encoder,
        }

        model = VqaLstmCnnAttentionModel(**model_kwargs)

        lossFn = torch.nn.CrossEntropyLoss()

        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.habitat_baselines.il.vqa.lr),
        )

        metrics = VqaMetric(
            info={"split": "train"},
            metric_names=[
                "loss",
                "accuracy",
                "mean_rank",
                "mean_reciprocal_rank",
            ],
            log_json=os.path.join(
                config.habitat_baselines.il.output_log_dir, "train.json"
            ),
        )

        t, epoch = 0, 1

        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_mean_rank = 0.0
        avg_mean_reciprocal_rank = 0.0

        logger.info(model)
        model.train().to(self.device)

        if config.habitat_baselines.il.vqa.freeze_encoder:
            model.cnn.eval()

        with TensorboardWriter(
            config.habitat_baselines.tensorboard_dir,
            flush_secs=self.flush_secs,
        ) as writer:
            while epoch <= config.habitat_baselines.il.vqa.max_epochs:
                start_time = time.time()
                for batch in train_loader:
                    t += 1
                    _, questions, answers, frame_queue = batch
                    optim.zero_grad()

                    questions = questions.to(self.device)
                    answers = answers.to(self.device)
                    frame_queue = frame_queue.to(self.device)

                    scores, _ = model(frame_queue, questions)
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

                    if t % config.habitat_baselines.log_interval == 0:
                        logger.info("Epoch: {}".format(epoch))
                        logger.info(metrics.get_stat_string())

                        writer.add_scalar("loss", metrics_loss, t)
                        writer.add_scalar("accuracy", accuracy, t)
                        writer.add_scalar("mean_rank", mean_rank, t)
                        writer.add_scalar(
                            "mean_reciprocal_rank", mean_reciprocal_rank, t
                        )

                        metrics.dump_log()

                # Dataloader length for IterableDataset doesn't take into
                # account batch size for Pytorch v < 1.6.0
                num_batches = math.ceil(
                    len(vqa_dataset)
                    / config.habitat_baselines.il.vqa.batch_size
                )

                avg_loss /= num_batches
                avg_accuracy /= num_batches
                avg_mean_rank /= num_batches
                avg_mean_reciprocal_rank /= num_batches

                end_time = time.time()
                time_taken = "{:.1f}".format((end_time - start_time) / 60)

                logger.info(
                    "Epoch {} completed. Time taken: {} minutes.".format(
                        epoch, time_taken
                    )
                )

                logger.info("Average loss: {:.2f}".format(avg_loss))
                logger.info("Average accuracy: {:.2f}".format(avg_accuracy))
                logger.info("Average mean rank: {:.2f}".format(avg_mean_rank))
                logger.info(
                    "Average mean reciprocal rank: {:.2f}".format(
                        avg_mean_reciprocal_rank
                    )
                )

                print("-----------------------------------------")

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

        with read_write(config):
            config.habitat.dataset.split = (
                self.config.habitat_baselines.eval.split
            )

        vqa_dataset = (
            EQADataset(
                config,
                input_type="vqa",
                num_frames=config.habitat_baselines.il.vqa.num_frames,
            )
            .shuffle(1000)
            .to_tuple(
                "episode_id",
                "question",
                "answer",
                *["{0:0=3d}.jpg".format(x) for x in range(0, 5)],
            )
            .map(img_bytes_2_np_array)
        )

        eval_loader = DataLoader(
            vqa_dataset, batch_size=config.habitat_baselines.il.vqa.batch_size
        )

        logger.info("eval_loader has {} samples".format(len(vqa_dataset)))

        q_vocab_dict, ans_vocab_dict = vqa_dataset.get_vocab_dicts()

        model_kwargs = {
            "q_vocab": q_vocab_dict.word2idx_dict,
            "ans_vocab": ans_vocab_dict.word2idx_dict,
            "eqa_cnn_pretrain_ckpt_path": config.habitat_baselines.il.eqa_cnn_pretrain_ckpt_path,
        }
        model = VqaLstmCnnAttentionModel(**model_kwargs)

        state_dict = torch.load(
            checkpoint_path, map_location={"cuda:0": "cpu"}
        )
        model.load_state_dict(state_dict)

        lossFn = torch.nn.CrossEntropyLoss()

        t = 0

        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_mean_rank = 0.0
        avg_mean_reciprocal_rank = 0.0

        model.eval()
        model.cnn.eval()
        model.to(self.device)

        metrics = VqaMetric(
            info={"split": "val"},
            metric_names=[
                "loss",
                "accuracy",
                "mean_rank",
                "mean_reciprocal_rank",
            ],
            log_json=os.path.join(
                config.habitat_baselines.il.output_log_dir, "eval.json"
            ),
        )
        with torch.no_grad():
            for batch in eval_loader:
                t += 1
                episode_ids, questions, answers, frame_queue = batch
                questions = questions.to(self.device)
                answers = answers.to(self.device)
                frame_queue = frame_queue.to(self.device)

                scores, _ = model(frame_queue, questions)

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

                if t % config.habitat_baselines.log_interval == 0:
                    logger.info(metrics.get_stat_string(mode=0))
                    metrics.dump_log()

                if (
                    config.habitat_baselines.il.eval_save_results
                    and t
                    % config.habitat_baselines.il.eval_save_results_interval
                    == 0
                ):
                    self._save_vqa_results(
                        checkpoint_index,
                        episode_ids,
                        questions,
                        frame_queue,
                        scores,
                        answers,
                        q_vocab_dict,
                        ans_vocab_dict,
                    )

        num_batches = math.ceil(
            len(vqa_dataset) / config.habitat_baselines.il.vqa.batch_size
        )

        avg_loss /= num_batches
        avg_accuracy /= num_batches
        avg_mean_rank /= num_batches
        avg_mean_reciprocal_rank /= num_batches

        writer.add_scalar("avg val loss", avg_loss, checkpoint_index)
        writer.add_scalar("avg val accuracy", avg_accuracy, checkpoint_index)
        writer.add_scalar("avg val mean rank", avg_mean_rank, checkpoint_index)
        writer.add_scalar(
            "avg val mean reciprocal rank",
            avg_mean_reciprocal_rank,
            checkpoint_index,
        )

        logger.info("Average loss: {:.2f}".format(avg_loss))
        logger.info("Average accuracy: {:.2f}".format(avg_accuracy))
        logger.info("Average mean rank: {:.2f}".format(avg_mean_rank))
        logger.info(
            "Average mean reciprocal rank: {:.2f}".format(
                avg_mean_reciprocal_rank
            )
        )
