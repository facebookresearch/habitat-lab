#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import OrderedDict
from typing import ClassVar, Dict, List

import cv2
import numpy as np
import torch

from habitat import Config, logger
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    poll_checkpoint_folder,
    tensor_to_depth_images,
    tensor_to_rgb_images,
)


class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
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

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                self._eval_checkpoint(self.config.EVAL_CKPT_PATH_DIR, writer)
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()
        config.defrost()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"

        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

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

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )


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
        if self.config.TRAINER_NAME == "edfe":
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

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                self._eval_checkpoint(self.config.EVAL_CKPT_PATH_DIR, writer)
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"

        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

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

    def _put_vqa_text_on_image(
        self,
        image: np.ndarray,
        question: str,
        prediction: str,
        ground_truth: str,
    ) -> np.ndarray:
        r"""For writing question, prediction and ground truth answer
            on image.

        Args:
            image: image on which text has to be written
            question: input question to model
            prediction: model's answer prediction
            ground_truth: ground truth answer

        Returns:
            image with text
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)
        scale = 0.4
        thickness = 1

        cv2.putText(
            image,
            "Question: " + question,
            (10, 15),
            font,
            scale,
            color,
            thickness,
        )
        cv2.putText(
            image,
            "Prediction: " + prediction,
            (10, 30),
            font,
            scale,
            color,
            thickness,
        )
        cv2.putText(
            image,
            "Ground truth: " + ground_truth,
            (10, 45),
            font,
            scale,
            color,
            thickness,
        )

        return image

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

        images = tensor_to_rgb_images(images_tensor)

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

        image = self._put_vqa_text_on_image(
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
        rgb_img, out_ae_img = tensor_to_rgb_images([rgb, out_ae])
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

        seg_img = seg.cpu().numpy()
        out_seg_img = torch.argmax(out_seg, 0).cpu().numpy()

        seg_img_color = self.colors[seg_img]
        out_seg_img_color = self.colors[out_seg_img]

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

    def _save_edfe_results(
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
        r"""For saving EDFE results.

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

        ckpt_epoch = ckpt_path[ckpt_path.rfind("/") + 1 :]
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
