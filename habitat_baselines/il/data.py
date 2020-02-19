import torch
from torch.utils.data import Dataset

import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from habitat.core.simulator import ShortestPathPoint


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

        # calculating max length of question tokens for padding later
        # can be done in mp3d_eqa_dataset while loading dataset
        self.max_q_len = 0
        for episode in self.episodes:
            if len(episode.question.question_tokens) > self.max_q_len:
                self.max_q_len = len(episode.question.question_tokens)

        ctr = 0

        for key in self.ans_vocab.word2idx_dict.keys():
            self.ans_vocab.word2idx_dict[key] = ctr
            ctr += 1

        self.scene_ids = []
        self.scene_episode_dict = {}

        # preparing a dict that stores list of episodes for each scene
        for ep in tqdm(self.episodes):
            if ep.scene_id not in self.scene_ids:
                self.scene_ids.append(ep.scene_id)
                self.scene_episode_dict[ep.scene_id] = [ep.episode_id]
            else:
                self.scene_episode_dict[ep.scene_id].append(ep.episode_id)

        # for each scene > load scene in memory > get last frames for each
        # episode corresponding to each scene
        for scene in tqdm(
            list(self.scene_episode_dict.keys()),
            desc="loading\
                          frames for each episode in memory",
        ):

            self.config.defrost()
            self.config.SIMULATOR.SCENE = scene
            self.config.freeze()
            self.env.sim.reconfigure(self.config.SIMULATOR)

            for ep_id in tqdm(self.scene_episode_dict[scene]):
                episode = next(
                    ep for ep in self.episodes if ep.episode_id == ep_id
                )
                pos_queue = episode.shortest_paths[0][-self.num_frames :]
                episode.frame_queue = self.get_frame_queue(pos_queue)

    def get_vocab_dicts(self) -> Tuple[dict, dict]:
        return self.q_vocab.word2idx_dict, self.ans_vocab.word2idx_dict

    def get_frame_queue(
        self, pos_queue: List[ShortestPathPoint], preprocess=True
    ) -> np.ndarray:
        frame_queue = []

        for pos in pos_queue:
            observation = self.env.sim.get_observations_at(
                pos.position, pos.rotation
            )
            img = observation["rgb"]

            if preprocess is True:
                img = img.transpose(2, 0, 1)
                img = img / 255.0

            frame_queue.append(img)

        return np.array(frame_queue)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Returns batch to trainer.

        example->
        actual question - "what color is the cabinet in the kitchen?"
        actual answer - "brown"

        batch ->
        question: [4,5,6,7,8,9,7,10,0,0..],
        answer: 2,
        frame_queue: list of images depending on trainer and model.

        """
        questions = self.episodes[idx].question.question_tokens
        answers = self.ans_vocab.word2idx(
            self.episodes[idx].question.answer_text
        )

        # padding question with zeros - to make all questions of same length
        # (can also be done in mp3d_eqa_dataset while loading dataset)
        if len(questions) < self.max_q_len:
            diff = self.max_q_len - len(questions)
            for i in range(diff):
                questions.append(0)

        frame_queue = self.episodes[idx].frame_queue
        questions = torch.LongTensor(questions)

        if self.input_type == "vqa":
            batch = idx, questions, answers, frame_queue

        return batch
