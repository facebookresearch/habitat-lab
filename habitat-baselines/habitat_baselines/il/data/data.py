# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
import webdataset as wds
import webdataset.filters as filters
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.core.simulator import ShortestPathPoint
from habitat.datasets.utils import VocabDict
from habitat_baselines.utils.common import (
    base_plus_ext,
    create_tar_archive,
    delete_folder,
    get_scene_episode_dict,
    valid_sample,
)


class EQADataset(wds.Dataset):
    """Pytorch dataset for Embodied Q&A (both VQA and PACMAN)"""

    def __init__(
        self,
        config,
        input_type,
        num_frames=5,
        max_controller_actions=5,
    ):
        """
        Args:
            config: Config
            input_type (string): Type of model being trained ("vqa", "pacman")
            num_frames (int): number of frames used as input to VQA model
            max_controller_actions (int):
        """
        self.config = config.habitat
        self.input_type = input_type
        self.num_frames = num_frames

        with habitat.Env(config=self.config) as self.env:
            self.episodes = self.env._dataset.episodes

            # sorting and making episode ids consecutive for simpler indexing
            self.sort_episodes()

            self.q_vocab = self.env._dataset.question_vocab
            self.ans_vocab = self.env._dataset.answer_vocab

            self.eval_save_results = config.habitat_baselines.eval_save_results

            if (
                self.config.dataset.split
                == config.habitat_baselines.eval.split
            ):
                self.mode = "val"
            else:
                self.mode = "train"

            self.frame_dataset_path = (
                config.habitat_baselines.frame_dataset_path.format(
                    split=self.mode
                )
            )

            # [TODO] can be done in mp3d_eqa_dataset when loading
            self.calc_max_length()
            self.restructure_ans_vocab()

            group_by_keys = filters.Curried(self.group_by_keys_)
            super().__init__(
                urls=self.frame_dataset_path + ".tar",
                initial_pipeline=[group_by_keys()],
            )

            self.only_vqa_task = config.habitat_baselines.only_vqa_task

            self.scene_episode_dict = get_scene_episode_dict(self.episodes)

            if not self.cache_exists():
                """
                for each scene > load scene in memory > save frames for each
                episode corresponding to each scene
                """
                logger.info(
                    "[ Dataset cache not present / is incomplete. ]\
                    \n[ Saving episode frames to disk. ]"
                )

                logger.info(
                    "Number of {} episodes: {}".format(
                        self.mode, len(self.episodes)
                    )
                )

                for scene in tqdm(
                    list(self.scene_episode_dict.keys()),
                    desc="Going through all scenes from dataset",
                ):
                    self.load_scene(scene)

                    for episode in tqdm(
                        self.scene_episode_dict[scene],
                        desc="Saving episode frames for each scene",
                    ):
                        if self.only_vqa_task:
                            pos_queue = episode.shortest_paths[0][
                                -self.num_frames :  # noqa: E203
                            ]
                        else:
                            pos_queue = episode.shortest_paths[0]

                        self.save_frame_queue(pos_queue, episode.episode_id)

                logger.info("[ Saved all episodes' frames to disk. ]")

                create_tar_archive(
                    self.frame_dataset_path + ".tar",
                    self.frame_dataset_path,
                )

                logger.info("[ Tar archive created. ]")
                logger.info(
                    "[ Deleting dataset folder. This will take a few minutes. ]"
                )
                delete_folder(self.frame_dataset_path)

                logger.info("[ Frame dataset is ready. ]")

    def group_by_keys_(
        self,
        data,
        keys: Callable[[str], Tuple[str, ...]] = base_plus_ext,
        lcase: bool = True,
        suffixes=None,
    ):
        """Returns function over iterator that groups key, value pairs into samples-
        a custom pipeline for grouping episode info & images in the webdataset.
        keys: function that splits the key into key and extension (base_plus_ext)
        lcase: convert suffixes to lower case (Default value = True)
        """
        current_sample: Dict[str, Any] = {}
        for fname, value in data:
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if not current_sample or prefix != current_sample["__key__"]:
                if valid_sample(current_sample):
                    yield current_sample

                current_sample = dict(__key__=prefix)

                episode_id = int(prefix[prefix.rfind("/") + 1 :])
                current_sample["episode_id"] = self.episodes[
                    episode_id
                ].episode_id

                question = self.episodes[episode_id].question.question_tokens
                if len(question) < self.max_q_len:
                    diff = self.max_q_len - len(question)
                    for _ in range(diff):
                        question.append(0)

                current_sample["question"] = torch.LongTensor(question)
                current_sample["answer"] = self.ans_vocab.word2idx(
                    self.episodes[episode_id].question.answer_text
                )
            if suffix in current_sample:
                raise ValueError(
                    f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
                )
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value

        if valid_sample(current_sample):
            yield current_sample

    def calc_max_length(self) -> None:
        r"""Calculates max length of questions and actions.
        This will be used for padding questions and actions with 0s so that
        they have same string length.
        """
        self.max_q_len = max(
            len(episode.question.question_tokens) for episode in self.episodes
        )
        self.max_action_len = max(
            len(episode.shortest_paths[0]) for episode in self.episodes
        )

    def restructure_ans_vocab(self) -> None:
        r"""
        Restructures answer vocab so that each answer id corresponds to a
        numerical index starting from 0 for first answer.
        """
        for idx, key in enumerate(sorted(self.ans_vocab.word2idx_dict.keys())):
            self.ans_vocab.word2idx_dict[key] = idx

    def get_vocab_dicts(self) -> Tuple[VocabDict, VocabDict]:
        r"""Returns Q&A VocabDicts"""
        return self.q_vocab, self.ans_vocab

    def sort_episodes(self) -> None:
        # TODO: can be done in mp3d_eqa_dataset class too?
        self.episodes = sorted(self.episodes, key=lambda x: int(x.episode_id))
        for idx, ep in enumerate(self.episodes):
            ep.episode_id = idx

    def save_frame_queue(
        self,
        pos_queue: List[ShortestPathPoint],
        episode_id,
    ) -> None:
        r"""Writes episode's frame queue to disk."""

        for idx, pos in enumerate(pos_queue[::-1]):
            observation = self.env.sim.get_observations_at(
                pos.position, pos.rotation
            )
            img = observation["rgb"]
            str_idx = "{0:0=3d}".format(idx)
            episode_id = "{0:0=4d}".format(int(episode_id))
            new_path = os.path.join(
                self.frame_dataset_path, "{}.{}".format(episode_id, str_idx)
            )
            cv2.imwrite(new_path + ".jpg", img[..., ::-1])

    def get_frames(self, frames_path, num=0):
        r"""Fetches frames from disk."""
        frames = []
        for img in sorted(os.listdir(frames_path))[-num:]:
            img_path = os.path.join(frames_path, img)
            img = cv2.imread(img_path)[..., ::-1]
            img = img.transpose(2, 0, 1)
            img = img / 255.0
            frames.append(img)
        return np.array(frames, dtype=np.float32)

    def cache_exists(self) -> bool:
        if os.path.exists(self.frame_dataset_path + ".tar"):
            return True
        else:
            os.makedirs(self.frame_dataset_path, exist_ok=True)
            return False

    def load_scene(self, scene) -> None:
        self.config.defrost()
        self.config.simulator.scene = scene
        self.config.freeze()
        self.env.sim.reconfigure(self.config.simulator)

    def __len__(self) -> int:
        return len(self.episodes)
