import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.core.simulator import ShortestPathPoint
from habitat.datasets.utils import VocabDict
from habitat_baselines.utils.common import get_scene_episode_dict


class EQADataset(Dataset):
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
        self.config = config.TASK_CONFIG
        self.env = habitat.Env(config=self.config)
        self.input_type = input_type
        self.num_frames = num_frames

        self.episodes = self.env._dataset.episodes

        self.q_vocab = self.env._dataset.question_vocab
        self.ans_vocab = self.env._dataset.answer_vocab

        self.eval_save_results = config.EVAL_SAVE_RESULTS

        if self.config.DATASET.SPLIT == config.EVAL.SPLIT:
            self.mode = "val"
            self.sort_eval_episodes()
        else:
            self.mode = "train"

        self.frame_dataset_path = config.FRAME_DATASET_PATH.format(
            split=self.mode
        )
        self.only_vqa_task = config.ONLY_VQA_TASK

        # [TODO] can be done in mp3d_eqa_dataset while loading dataset
        self.calc_max_length()
        self.restructure_ans_vocab()

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

            logger.info(
                "[ Saved all episodes' frames to disk. Frame dataset ready. ]"
            )

        self.env.close()

    def calc_max_length(self) -> None:
        r"""Calculates max length of questions and actions.
        This will be used for padding questions and actions with 0s so that
        they have same string length.
        """
        self.max_q_len = 0
        self.max_action_len = 0

        for episode in self.episodes:
            if len(episode.question.question_tokens) > self.max_q_len:
                self.max_q_len = len(episode.question.question_tokens)

            if len(episode.shortest_paths[0]) > self.max_action_len:
                self.max_action_len = len(episode.shortest_paths[0])

    def restructure_ans_vocab(self) -> None:
        r"""
        Restructures answer vocab so that each answer id corresponds to a
        numerical index starting from 0 for first answer.
        """

        ctr = 0
        for key in self.ans_vocab.word2idx_dict.keys():
            self.ans_vocab.word2idx_dict[key] = ctr
            ctr += 1

    def get_vocab_dicts(self) -> Tuple[VocabDict, VocabDict]:
        r"""Returns Q&A VocabDicts"""
        return self.q_vocab, self.ans_vocab

    def sort_eval_episodes(self) -> None:
        self.episodes = sorted(self.episodes, key=lambda x: int(x.episode_id))
        for idx, ep in enumerate(self.episodes):
            ep.episode_id = idx

    def save_frame_queue(
        self,
        pos_queue: List[ShortestPathPoint],
        episode_id,
    ) -> None:
        r"""Writes episode's frame queue to disk."""
        episode_frames_path = os.path.join(
            self.frame_dataset_path, str(episode_id)
        )
        if not os.path.exists(episode_frames_path):
            os.makedirs(episode_frames_path)

        for idx, pos in enumerate(pos_queue):
            observation = self.env.sim.get_observations_at(
                pos.position, pos.rotation
            )
            img = observation["rgb"]
            frame_path = os.path.join(
                episode_frames_path, "{0:0=3d}".format(idx)
            )
            cv2.imwrite(frame_path + ".jpg", img)

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
        if os.path.exists(self.frame_dataset_path):
            if len(os.listdir(self.frame_dataset_path)) == len(self.episodes):
                return True
        else:
            os.makedirs(self.frame_dataset_path)
        return False

    def load_scene(self, scene) -> None:
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.freeze()
        self.env.sim.reconfigure(self.config.SIMULATOR)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        r"""Returns batch to trainer.

        example->
        actual question - "what color is the cabinet in the kitchen?"
        actual answer - "brown"

        batch ->
        question: [4,5,6,7,8,9,7,10,0,0..],
        answer: 2,
        frame_queue: tensor containing episode frames

        """

        episode_id = self.episodes[idx].episode_id
        question = self.episodes[idx].question.question_tokens
        answer = self.ans_vocab.word2idx(
            self.episodes[idx].question.answer_text
        )

        # padding question with zeros - to make all questions of same length
        if len(question) < self.max_q_len:
            diff = self.max_q_len - len(question)
            for _ in range(diff):
                question.append(0)

        question = torch.LongTensor(question)
        frames_path = os.path.join(self.frame_dataset_path, str(episode_id))

        if self.input_type == "vqa":
            frame_queue = self.get_frames(frames_path, num=self.num_frames)
            batch = idx, question, answer, frame_queue
            return batch
