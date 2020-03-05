import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from habitat.core.simulator import ShortestPathPoint
from habitat_baselines.il.models.models import MultitaskCNN


class EQADataset(Dataset):
    """Pytorch dataset for Embodied Q&A (both VQA and NAV)"""

    def __init__(
        self, env, config, input_type, num_frames=5, max_controller_actions=5,
    ):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            input_type (string): Type of model being trained ("vqa", "pacman")
            num_frames (int): number of frames used as input to VQA model
            max_controller_actions (int):
        """
        self.env = env
        self.config = config.TASK_CONFIG
        self.input_type = input_type
        self.num_frames = num_frames

        self.sim = self.env.sim
        self.episodes = self.env._dataset.episodes

        self.q_vocab = self.env._dataset.question_vocab
        self.ans_vocab = self.env._dataset.answer_vocab

        self.eval_save_results = config.EVAL_SAVE_RESULTS

        if self.config.DATASET.SPLIT == config.EVAL.SPLIT:
            self.mode = "val"
        else:
            self.mode = "train"

        self.frame_dataset_path = config.FRAME_DATASET_PATH
        self.only_vqa_task = config.ONLY_VQA_TASK
        self.disk_cache_exists = False

        # [TODO] can be done in mp3d_eqa_dataset while loading dataset
        self.calc_max_length()
        self.restructure_ans_vocab()

        self.scene_ids = []
        self.scene_episode_dict = {}

        # preparing a dict that stores list of episodes for each scene
        for ep in tqdm(self.episodes):
            if ep.scene_id not in self.scene_ids:
                self.scene_ids.append(ep.scene_id)
                self.scene_episode_dict[ep.scene_id] = [ep.episode_id]
            else:
                self.scene_episode_dict[ep.scene_id].append(ep.episode_id)

        # checking if cache exists & making cache dir
        if not os.path.exists(os.path.join(self.frame_dataset_path, self.mode)):
            os.makedirs(os.path.join(self.frame_dataset_path, self.mode))
            print('Disk cache does not exist.')

        else:
            if len(os.listdir(os.path.join(self.frame_dataset_path, self.mode))) == len(self.episodes):
                self.disk_cache_exists = True
                print('Disk cache exists.')

        if not self.disk_cache_exists:
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to each scene
            """
            print('Saving episode frames to disk.')
            for scene in tqdm(
                list(self.scene_episode_dict.keys()),
                desc="going through all scenes from dataset"
            ):

                self.config.defrost()
                self.config.SIMULATOR.SCENE = scene
                self.config.freeze()
                self.env.sim.reconfigure(self.config.SIMULATOR)

                for ep_id in tqdm(self.scene_episode_dict[scene],
                                  desc="saving episode frames for each scene"):
                    episode = next(
                        ep for ep in self.episodes if ep.episode_id == ep_id
                    )
                    if self.only_vqa_task:
                        pos_queue = episode.shortest_paths[0][-self.num_frames:]
                    else:
                        pos_queue = episode.shortest_paths[0]

                    self.save_frame_queue(pos_queue, ep_id, self.mode)

        print('Saved all episodes\' frames to disk. Frame dataset ready.')
        self.env.close()

    def calc_max_length(self) -> None:
        r"""Calculates max length of question.
        This will be used for padding questions with 0s so that all questions
        have same string length.
        """
        self.max_q_len = 0

        for episode in self.episodes:
            if len(episode.question.question_tokens) > self.max_q_len:
                self.max_q_len = len(episode.question.question_tokens)

    def restructure_ans_vocab(self) -> None:
        r"""
        Restructures answer vocab so that each answer id corresponds to a
        numerical index starting from 0 for first answer.
        """

        ctr = 0
        for key in self.ans_vocab.word2idx_dict.keys():
            self.ans_vocab.word2idx_dict[key] = ctr
            ctr += 1

    def get_vocab_dicts(self) -> Tuple[dict, dict]:
        r"""Returns vocab dictionaries from vocabs

        """
        return self.q_vocab.word2idx_dict, self.ans_vocab.word2idx_dict

    def save_frame_queue(
        self, pos_queue: List[ShortestPathPoint], episode_id, mode
    ) -> None:
        r"""Writes episode's frame queue to disk.
        """
        episode_frames_path = os.path.join(self.frame_dataset_path, mode, episode_id)
        if not os.path.exists(episode_frames_path):
            os.makedirs(episode_frames_path)

        for idx, pos in enumerate(pos_queue):
            observation = self.env.sim.get_observations_at(
                pos.position, pos.rotation
            )
            img = observation["rgb"]
            frame_path = os.path.join(episode_frames_path, "{0:0=3d}".format(idx))
            cv2.imwrite(frame_path + '.jpg', img)

    def get_frames(self, frames_path, num=0):
        r"""Fetches frames from disk.
        """
        frames = []
        for img in sorted(os.listdir(frames_path))[-num:]:
            img_path = os.path.join(frames_path, img)
            img = cv2.imread(img_path)
            img = img.transpose(2, 0, 1)
            img = img / 255.0
            frames.append(img)
        return np.array(frames)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(
        self, idx: int
    ):
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
            for i in range(diff):
                question.append(0)

        question = torch.LongTensor(question)
        frames_path = os.path.join(self.frame_dataset_path, self.mode, episode_id)

        if self.input_type == "vqa":
            frame_queue = self.get_frames(frames_path, num=self.num_frames)
            batch = idx, question, answer, frame_queue
        return batch
