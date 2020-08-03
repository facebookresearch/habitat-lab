#!/usr/bin/env python3

import os
from collections import OrderedDict
from typing import Dict, List

import cv2
import numpy as np
import torch

from habitat import Config
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    tensor_to_bgr_images,
    tensor_to_depth_images,
    put_vqa_text_on_image,
)
from habitat_sim.utils.common import d3_40_colors_rgb


class BaseILTrainer(BaseTrainer):
    r"""Base trainer class for IL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device
    config: Config
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30
        self._make_dirs()

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def _make_dirs(self):
        self._make_log_dir()
        self._make_ckpt_dir()
        if self.config.EVAL_SAVE_RESULTS:
            self._make_results_dir()

    def _make_log_dir(self):
        if self.config.LOG_METRICS:
            if not os.path.isdir(self.config.OUTPUT_LOG_DIR):
                os.makedirs(self.config.OUTPUT_LOG_DIR)

    def _make_ckpt_dir(self):
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

    def _make_results_dir(self):
        if self.config.TRAINER_NAME == "eqa-cnn-pretrain":
            for type in ["rgb", "seg", "depth"]:
                dir_name = self.config.RESULTS_DIR.format(
                    split="val", type=type
                )
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
        else:
            dir_name = self.config.RESULTS_DIR.format(split="val")
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, state_dict: OrderedDict, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            state_dict: model's state_dict
            file_name: file name for checkpoint

        Returns:
            None
        """
        torch.save(
            state_dict, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    def _get_q_string(self, question: List, q_vocab_dict: Dict) -> str:
        r"""
        Converts question tokens to question string.
        """
        q_string = ""
        for token in question:
            if token != 0:
                for word, idx in q_vocab_dict.items():
                    if idx == token:
                        q_word = word
                        break
                q_string += q_word + " "
            else:
                break
        q_string += "?"

        return q_string

    def _save_image_results(
        self,
        ckpt_idx: int,
        idx: int,
        images_tensor: torch.Tensor,
        question: str,
        prediction: str,
        ground_truth: str,
    ) -> None:
        r"""For saving image results.

        Args:
            idx: index of batch
            ckpt_idx: idx of ckpt being evaluated
            images_tensor: images' tensor containing input frames
            question: input question to model
            prediction: model's answer prediction
            ground_truth: ground truth answer

        Returns:
            None
        """
        path = self.config.RESULTS_DIR.format(
            split=self.config.TASK_CONFIG.DATASET.SPLIT
        )

        images = tensor_to_bgr_images(images_tensor)

        collage_image = cv2.hconcat(images)
        collage_image = cv2.copyMakeBorder(
            collage_image,
            55,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        image = put_vqa_text_on_image(
            collage_image, question, prediction, ground_truth
        )

        cv2.imwrite(
            os.path.join(path, "c_{}_{}_image.jpg".format(ckpt_idx, idx)),
            image,
        )

    def _save_vqa_results(
        self,
        ckpt_idx: int,
        idx: torch.Tensor,
        questions: torch.Tensor,
        images: torch.Tensor,
        scores: torch.Tensor,
        answers: torch.Tensor,
        q_vocab_dict: Dict,
        ans_vocab_dict: Dict,
    ) -> None:

        r"""For saving VQA results.

        Args:
            ckpt_idx: idx of checkpoint being evaluated
            idx: index of batch
            images_tensor: images' tensor containing input frames
            question: input question to model
            prediction: model's answer prediction
            ground_truth: ground truth answer

        Returns:
            None
        """
        idx = idx[0].item()
        question = questions[0]
        images = images[0]
        answer = answers[0]
        scores = scores[0]
        q_string = self._get_q_string(question, q_vocab_dict)

        value, index = scores.max(0)
        prediction = list(ans_vocab_dict.keys())[index]
        ground_truth = list(ans_vocab_dict.keys())[answer]

        print("Question: ", q_string)
        print("Predicted answer:", prediction)
        print("Ground-truth answer:", ground_truth)

        self._save_image_results(
            idx, ckpt_idx, images, q_string, prediction, ground_truth
        )

    def _save_rgb_results(
        self, rgb: torch.Tensor, out_ae: torch.Tensor
    ) -> None:
        r"""For saving RGB reconstruction results.

        Args:
            rgb: ground truth RGB image
            out_ae: ouput of autoencoder
        """
        rgb_path = self.results_path.format(split="val", type="rgb")
        rgb_img, out_ae_img = tensor_to_bgr_images([rgb, out_ae])
        cv2.imwrite(
            os.path.join(rgb_path, self.result_id + "_gt.jpg"), rgb_img
        )
        cv2.imwrite(
            os.path.join(rgb_path, self.result_id + "_output.jpg"), out_ae_img
        )

    def _save_seg_results(
        self, seg: torch.Tensor, out_seg: torch.Tensor
    ) -> None:
        r"""For saving Segmentation results.

        Args:
            seg: ground truth segmentation
            out_seg: ouput segmentation
        """

        seg_path = self.results_path.format(split="val", type="seg")

        seg_img = seg.cpu().numpy() % 40
        out_seg_img = torch.argmax(out_seg, 0).cpu().numpy() % 40

        seg_img_color = d3_40_colors_rgb[seg_img]
        out_seg_img_color = d3_40_colors_rgb[out_seg_img]

        cv2.imwrite(
            os.path.join(seg_path, self.result_id + "_gt.jpg"), seg_img_color
        )
        cv2.imwrite(
            os.path.join(seg_path, self.result_id + "_output.jpg"),
            out_seg_img_color,
        )

    def _save_depth_results(
        self, depth: torch.Tensor, out_depth: torch.Tensor
    ) -> None:
        r"""For saving depth results.

        Args:
            depth: ground truth depth map
            out_depth: ouput depth map
        """
        depth_path = self.results_path.format(split="val", type="depth")

        depth_img, out_depth_img = tensor_to_depth_images([depth, out_depth])

        cv2.imwrite(
            os.path.join(depth_path, self.result_id + "_gt.jpg"), depth_img
        )
        cv2.imwrite(
            os.path.join(depth_path, self.result_id + "_output.jpg"),
            out_depth_img,
        )

    def _save_eqa_cnn_pretrain_results(
        self,
        ckpt_idx: int,
        idx: torch.Tensor,
        rgb: torch.Tensor,
        out_ae: torch.Tensor,
        seg: torch.Tensor,
        out_seg: torch.Tensor,
        depth: torch.Tensor,
        out_depth: torch.Tensor,
    ) -> None:
        r"""For saving EQA-CNN-Pretrained model's results.

        Args:
            ckpt_idx: index of ckpt being evaluated
            idx: batch index
            rgb: rgb ground truth
            out_ae: autoencoder output rgb reconstruction
            seg: segmentation ground truth
            out_seg: segmentation output
            depth: depth map ground truth
            out_depth: depth map output
        """

        self.results_path = self.config.RESULTS_DIR

        self.result_id = "c_{}_{}".format(ckpt_idx, idx[0].item())

        self._save_rgb_results(rgb[0], out_ae[0])
        self._save_seg_results(seg[0], out_seg[0])
        self._save_depth_results(depth[0], out_depth[0])

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

        ckpt_epoch = os.path.basename(ckpt_path)
        results_dir = os.path.join(results_dir, ckpt_epoch)

        q_string = self._get_q_string(question, q_vocab_dict)

        for idx, img in enumerate(imgs):
            border_width = 32
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 0, 0)
            scale = 0.3
            thickness = 1

            img = cv2.copyMakeBorder(
                img,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

            img = cv2.putText(
                img,
                "Question: " + q_string,
                (10, 15),
                font,
                scale,
                color,
                thickness,
            )

            imgs[idx] = img

        images_to_video(imgs, results_dir, "ep_" + str(t))
