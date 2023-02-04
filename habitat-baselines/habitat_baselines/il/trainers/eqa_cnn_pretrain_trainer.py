#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import torch
from torch.utils.data import DataLoader

from habitat import logger
from habitat.config import read_write
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
    EQACNNPretrainDataset,
)
from habitat_baselines.il.models.models import MultitaskCNN
from habitat_baselines.utils.visualizations.utils import (
    save_depth_results,
    save_rgb_results,
    save_seg_results,
)


@baseline_registry.register_trainer(name="eqa-cnn-pretrain")
class EQACNNPretrainTrainer(BaseILTrainer):
    r"""Trainer class for Encoder-Decoder for Feature Extraction
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

    def _make_results_dir(self):
        r"""Makes directory for saving eqa-cnn-pretrain eval results."""
        for s_type in ["rgb", "seg", "depth"]:
            dir_name = self.config.habitat_baselines.il.results_dir.format(
                split="val", type=s_type
            )
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

    def _save_results(
        self,
        gt_rgb: torch.Tensor,
        pred_rgb: torch.Tensor,
        gt_seg: torch.Tensor,
        pred_seg: torch.Tensor,
        gt_depth: torch.Tensor,
        pred_depth: torch.Tensor,
        path: str,
    ) -> None:
        r"""For saving EQA-CNN-Pretrain reconstruction results.

        Args:
            gt_rgb: rgb ground truth
            preg_rgb: autoencoder output rgb reconstruction
            gt_seg: segmentation ground truth
            pred_seg: segmentation output
            gt_depth: depth map ground truth
            pred_depth: depth map output
            path: to write file
        """

        save_rgb_results(gt_rgb[0], pred_rgb[0], path)
        save_seg_results(gt_seg[0], pred_seg[0], path)
        save_depth_results(gt_depth[0], pred_depth[0], path)

    def train(self) -> None:
        r"""Main method for pre-training Encoder-Decoder Feature Extractor for EQA.

        Returns:
            None
        """
        config = self.config

        eqa_cnn_pretrain_dataset = EQACNNPretrainDataset(config)

        train_loader = DataLoader(
            eqa_cnn_pretrain_dataset,
            batch_size=config.habitat_baselines.il.eqa_cnn_pretrain.batch_size,
            shuffle=True,
        )

        logger.info(
            "[ train_loader has {} samples ]".format(
                len(eqa_cnn_pretrain_dataset)
            )
        )

        model = MultitaskCNN()
        model.train().to(self.device)

        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.habitat_baselines.il.eqa_cnn_pretrain.lr),
        )

        depth_loss = torch.nn.SmoothL1Loss()
        ae_loss = torch.nn.SmoothL1Loss()
        seg_loss = torch.nn.CrossEntropyLoss()

        epoch, t = 1, 0
        with TensorboardWriter(
            config.habitat_baselines.tensorboard_dir,
            flush_secs=self.flush_secs,
        ) as writer:
            while (
                epoch
                <= config.habitat_baselines.il.eqa_cnn_pretrain.max_epochs
            ):
                start_time = time.time()
                avg_loss = 0.0

                for batch in train_loader:
                    t += 1

                    idx, gt_rgb, gt_depth, gt_seg = batch

                    optim.zero_grad()

                    gt_rgb = gt_rgb.to(self.device)
                    gt_depth = gt_depth.to(self.device)
                    gt_seg = gt_seg.to(self.device)

                    pred_seg, pred_depth, pred_rgb = model(gt_rgb)

                    l1 = seg_loss(pred_seg, gt_seg.long())
                    l2 = ae_loss(pred_rgb, gt_rgb)
                    l3 = depth_loss(pred_depth, gt_depth)

                    loss = l1 + (10 * l2) + (10 * l3)

                    avg_loss += loss.item()

                    if t % config.habitat_baselines.log_interval == 0:
                        logger.info(
                            "[ Epoch: {}; iter: {}; loss: {:.3f} ]".format(
                                epoch, t, loss.item()
                            )
                        )

                        writer.add_scalar("loss/total_loss", loss, t)
                        writer.add_scalar("loss/seg_loss", l1, t)
                        writer.add_scalar("loss/ae_loss", l2, t)
                        writer.add_scalar("loss/depth_loss", l3, t)

                    loss.backward()
                    optim.step()

                end_time = time.time()
                time_taken = "{:.1f}".format((end_time - start_time) / 60)
                avg_loss = avg_loss / len(train_loader)

                logger.info(
                    "[ Epoch {} completed. Time taken: {} minutes. ]".format(
                        epoch, time_taken
                    )
                )
                logger.info("[ Average loss: {:.3f} ]".format(avg_loss))

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

        eqa_cnn_pretrain_dataset = EQACNNPretrainDataset(config, mode="val")

        eval_loader = DataLoader(
            eqa_cnn_pretrain_dataset,
            batch_size=config.habitat_baselines.il.eqa_cnn_pretrain.batch_size,
            shuffle=False,
        )

        logger.info(
            "[ eval_loader has {} samples ]".format(
                len(eqa_cnn_pretrain_dataset)
            )
        )

        model = MultitaskCNN()

        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

        model.to(self.device).eval()

        depth_loss = torch.nn.SmoothL1Loss()
        ae_loss = torch.nn.SmoothL1Loss()
        seg_loss = torch.nn.CrossEntropyLoss()

        t = 0
        avg_loss = 0.0
        avg_l1 = 0.0
        avg_l2 = 0.0
        avg_l3 = 0.0

        with torch.no_grad():
            for batch in eval_loader:
                t += 1

                idx, gt_rgb, gt_depth, gt_seg = batch
                gt_rgb = gt_rgb.to(self.device)
                gt_depth = gt_depth.to(self.device)
                gt_seg = gt_seg.to(self.device)

                pred_seg, pred_depth, pred_rgb = model(gt_rgb)
                l1 = seg_loss(pred_seg, gt_seg.long())
                l2 = ae_loss(pred_rgb, gt_rgb)
                l3 = depth_loss(pred_depth, gt_depth)

                loss = l1 + (10 * l2) + (10 * l3)

                avg_loss += loss.item()
                avg_l1 += l1.item()
                avg_l2 += l2.item()
                avg_l3 += l3.item()

                if t % config.habitat_baselines.log_interval == 0:
                    logger.info(
                        "[ Iter: {}; loss: {:.3f} ]".format(t, loss.item()),
                    )

                if (
                    config.habitat_baselines.il.eval_save_results
                    and t
                    % config.habitat_baselines.il.eval_save_results_interval
                    == 0
                ):
                    result_id = "ckpt_{}_{}".format(
                        checkpoint_index, idx[0].item()
                    )
                    result_path = os.path.join(
                        self.config.habitat_baselines.il.results_dir, result_id
                    )

                    self._save_results(
                        gt_rgb,
                        pred_rgb,
                        gt_seg,
                        pred_seg,
                        gt_depth,
                        pred_depth,
                        result_path,
                    )

        avg_loss /= len(eval_loader)
        avg_l1 /= len(eval_loader)
        avg_l2 /= len(eval_loader)
        avg_l3 /= len(eval_loader)

        writer.add_scalar(
            "avg_val_loss/total_loss", avg_loss, checkpoint_index
        )
        writer.add_scalar("avg_val_loss/seg_loss", avg_l1, checkpoint_index)
        writer.add_scalar("avg_val_loss/ae_loss", avg_l2, checkpoint_index)
        writer.add_scalar("avg_val_loss/depth_loss", avg_l3, checkpoint_index)

        logger.info("[ Average loss: {:.3f} ]".format(avg_loss))
        logger.info("[ Average seg loss: {:.3f} ]".format(avg_l1))
        logger.info("[ Average autoencoder loss: {:.4f} ]".format(avg_l2))
        logger.info("[ Average depthloss: {:.4f} ]".format(avg_l3))
