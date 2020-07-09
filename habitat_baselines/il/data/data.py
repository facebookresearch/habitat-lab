import copy
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from habitat import logger
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
        # first scene that is loaded by default
        self.temp_scene_id = self.episodes[0].scene_id

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

        if self.input_type == "pacman":

            cnn_kwargs = {
                "num_classes": 40,
                "pretrained": True,
                "checkpoint_path": config.EQA_CNN_PRETRAIN_CKPT_PATH,
            }
            self.cnn = MultitaskCNN(**cnn_kwargs)
            self.cnn.eval()
            self.cnn.cuda()

            self.max_controller_actions = max_controller_actions
            self.preprocess_actions()

        if self.mode == "val":
            """
            For eval, we create a new episodes list that is grouped based
            on the scene-ids. This allows us to load a scene once, and
            use that for all of the eval episodes corresponding to that
            scene id, then load another scene and so on.
            """
            self.eval_episodes = []
            ctr = 0
            for scene in tqdm(
                list(self.scene_episode_dict.keys()),
                desc="going through all scenes from dataset",
            ):
                for ep_id in self.scene_episode_dict[scene]:
                    episode = next(
                        ep for ep in self.episodes if ep.episode_id == ep_id
                    )

                    tmp_episode = copy.copy(episode)
                    tmp_episode.episode_id = ctr
                    self.eval_episodes.append(tmp_episode)
                    ctr += 1

        # checking if cache exists & making cache dir
        if not os.path.exists(
            os.path.join(self.frame_dataset_path, self.mode)
        ):
            os.makedirs(os.path.join(self.frame_dataset_path, self.mode))

        else:
            if len(
                os.listdir(os.path.join(self.frame_dataset_path, self.mode))
            ) == len(self.episodes):
                self.disk_cache_exists = True
                logger.info("[ Disk cache exists. ]")

        if not self.disk_cache_exists:
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to each scene
            """

            logger.info(
                "[ Disk cache not present / isincomplete. ]\
                \n[ Saving episode frames to disk. ]"
            )

            ctr = 0

            for scene in tqdm(
                list(self.scene_episode_dict.keys()),
                desc="Going through all scenes from dataset",
            ):

                self.config.defrost()
                self.config.SIMULATOR.SCENE = scene
                self.config.freeze()
                self.env.sim.reconfigure(self.config.SIMULATOR)

                for ep_id in tqdm(
                    self.scene_episode_dict[scene],
                    desc="Saving episode frames for each scene",
                ):
                    episode = next(
                        ep for ep in self.episodes if ep.episode_id == ep_id
                    )

                    if self.only_vqa_task:
                        pos_queue = episode.shortest_paths[0][
                            -self.num_frames :  # noqa: E203
                        ]
                    else:
                        pos_queue = episode.shortest_paths[0]

                    if self.mode == "val":
                        self.save_frame_queue(pos_queue, ctr, self.mode)
                        ctr += 1
                    else:
                        self.save_frame_queue(pos_queue, ep_id, self.mode)

            logger.info(
                "[ Saved all episodes' frames to disk. Frame dataset ready. ]"
            )

        if self.input_type != "pacman" and self.mode != "val":
            self.env.close()

    def calc_max_length(self) -> None:
        r"""Calculates max length of questions and actions.
        This will be used for padding questions and actions so that they
        have same string length.
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

    def get_vocab_dicts(self) -> Tuple[dict, dict]:
        r"""Returns vocab dictionaries from vocabs

        """
        return self.q_vocab.word2idx_dict, self.ans_vocab.word2idx_dict

    def save_frame_queue(
        self, pos_queue: List[ShortestPathPoint], episode_id, mode
    ) -> None:
        r"""Writes episode's frame queue to disk.
        """
        episode_frames_path = os.path.join(
            self.frame_dataset_path, mode, str(episode_id)
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
            cv2.imwrite(frame_path + ".jpg", img[..., ::-1])

    def get_frames(self, frames_path, num=0):
        r"""Fetches frames from disk.
        """
        frames = []
        for img in sorted(os.listdir(frames_path))[-num:]:
            img_path = os.path.join(frames_path, img)
            img = cv2.imread(img_path)[..., ::-1]
            img = img.transpose(2, 0, 1)
            img = img / 255.0
            frames.append(img)
        return np.array(frames, dtype=np.float32)

    """
    if the action sequence is [f, f, l, l, f, f, f, r]

    input sequence to planner is [<start>, f, l, f, r]
    output sequence for planner is [f, l, f, r, <end>]

    input sequences to controller are [f, f, l, l, f, f, f, r]
    output sequences for controller are [1, 0, 1, 0, 1, 1, 0, 0]
    """

    def flat_to_hierarchical_actions(self, actions, controller_action_lim):
        assert len(actions) != 0

        controller_action_ctr = 0

        # actions: [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 3, 3, 3, 3, 3, 1, 4]
        # planner_actions: [1, 2]
        planner_actions, controller_actions = [1], []
        prev_action = 1

        pq_idx, cq_idx, ph_idx = [], [], []
        ph_trck = 0

        for i in range(len(actions)):
            if actions[i] != prev_action:
                planner_actions.append(actions[i])
                pq_idx.append(i)

            if i > 0:
                ph_idx.append(ph_trck)
                if actions[i] == prev_action:
                    controller_actions.append(1)
                    controller_action_ctr += 1
                else:
                    controller_actions.append(0)
                    controller_action_ctr = 0
                    ph_trck += 1
                cq_idx.append(i)

            prev_action = actions[i]

            if controller_action_ctr == controller_action_lim - 1:
                prev_action = False

        return planner_actions, controller_actions, pq_idx, cq_idx, ph_idx

    def preprocess_actions(self) -> None:
        """
        actions before -
        0 - FWD; 1 - LEFT; 2 - RIGHT; 3 - STOP;
        actions after -
        0 - NULL; 1 - START; 2 - FWD; 3 - LEFT; 4 - RIGHT; 5 - STOP;
        """
        for ep in self.episodes:
            ep.actions = [x.action + 2 for x in ep.shortest_paths[0]]
            ep.action_length = len(ep.actions)

            pa, ca, pq_idx, cq_idx, ph_idx = self.flat_to_hierarchical_actions(
                actions=ep.actions,
                controller_action_lim=self.max_controller_actions,
            )

            # padding actions with 0
            if ep.action_length < self.max_action_len:
                diff = self.max_action_len - ep.action_length
                for i in range(diff):
                    ep.actions.append(0)

            ep.actions = torch.Tensor(ep.actions)
            ep.planner_actions = ep.actions.clone().fill_(0)
            ep.controller_actions = ep.actions.clone().fill_(-1)

            ep.planner_hidden_idx = ep.actions.clone().fill_(0)
            ep.planner_pos_queue_idx, ep.controller_pos_queue_idx = [], []

            ep.planner_actions[: len(pa)] = torch.Tensor(pa)
            ep.controller_actions[: len(ca)] = torch.Tensor(ca)

            ep.planner_action_length = len(pa) - 1
            ep.controller_action_length = len(ca)

            ep.planner_pos_queue_idx.append(pq_idx)
            ep.controller_pos_queue_idx.append(cq_idx)

            ep.planner_hidden_idx[: len(ca)] = torch.Tensor(ph_idx)

    def get_hierarchical_features_till_spawn(
        self, idx, actions, backtrack_steps=0, max_controller_actions=5
    ):

        action_length = len(actions)

        pa, ca, pq_idx, cq_idx, ph_idx = self.flat_to_hierarchical_actions(
            actions=actions, controller_action_lim=max_controller_actions
        )

        # count how many actions of same type have been encountered before
        # starting navigation

        backtrack_controller_steps = actions[
            0 : action_length - backtrack_steps + 1 :  # noqa: E203
        ][::-1]
        counter = 0

        if len(backtrack_controller_steps) > 0:
            while (counter <= self.max_controller_actions) and (
                counter < len(backtrack_controller_steps)
                and (
                    backtrack_controller_steps[counter]
                    == backtrack_controller_steps[0]
                )
            ):
                counter += 1

        target_pos_idx = action_length - backtrack_steps

        controller_step = True
        if target_pos_idx in pq_idx:
            controller_step = False

        pq_idx_pruned = [v for v in pq_idx if v <= target_pos_idx]
        pa_pruned = pa[: len(pq_idx_pruned) + 1]

        frames_path = os.path.join(
            self.frame_dataset_path, self.mode, str(idx)
        )

        images = self.get_frames(frames_path)
        raw_img_feats = (
            self.cnn(torch.FloatTensor(images).cuda())
            .data.cpu()
            .numpy()
            .copy()
        )

        controller_img_feat = torch.from_numpy(
            raw_img_feats[target_pos_idx].copy()
        )
        controller_action_in = pa_pruned[-1] - 2

        planner_img_feats = torch.from_numpy(
            raw_img_feats[pq_idx_pruned].copy()
        )
        planner_actions_in = torch.from_numpy(np.array(pa_pruned[:-1]) - 1)

        init_pos = self.eval_episodes[idx].shortest_paths[0][target_pos_idx]

        return (
            planner_actions_in,
            planner_img_feats,
            controller_step,
            controller_action_in,
            controller_img_feat,
            init_pos,
            counter,
        )

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

        if self.mode == "val":
            self.episodes = self.eval_episodes

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
        frames_path = os.path.join(
            self.frame_dataset_path, self.mode, str(episode_id)
        )

        if self.input_type == "vqa":
            frame_queue = self.get_frames(frames_path, num=self.num_frames)
            batch = idx, question, answer, frame_queue
            return batch

        elif self.input_type == "pacman":

            if self.mode == "val":
                # will work only with batch size 1

                actions = self.eval_episodes[idx].actions
                action_length = self.eval_episodes[idx].action_length
                scene = self.eval_episodes[idx].scene_id

                if scene != self.temp_scene_id:
                    logger.info("[ Loading scene - {}]".format(scene))
                    self.temp_scene_id = scene
                    self.config.defrost()
                    self.config.SIMULATOR.SCENE = scene
                    self.config.freeze()
                    self.env.sim.reconfigure(self.config.SIMULATOR)

                goal_pos = self.eval_episodes[idx].goals[0].position

                return idx, question, answer, actions, action_length, goal_pos

            planner_actions = self.episodes[idx].planner_actions
            controller_actions = self.episodes[idx].controller_actions

            planner_hidden_idx = self.episodes[idx].planner_hidden_idx

            planner_action_length = self.episodes[idx].planner_action_length
            controller_action_length = self.episodes[
                idx
            ].controller_action_length

            frame_queue = self.get_frames(frames_path)

            raw_img_feats = (
                self.cnn(torch.FloatTensor(frame_queue).cuda())
                .data.cpu()
                .numpy()
                .copy()
            )
            img_feats = np.zeros(
                (self.max_action_len, raw_img_feats.shape[1]), dtype=np.float32
            )
            img_feats[: raw_img_feats.shape[0], :] = raw_img_feats.copy()

            planner_pos_queue_idx = self.episodes[idx].planner_pos_queue_idx
            controller_pos_queue_idx = self.episodes[
                idx
            ].controller_pos_queue_idx

            planner_img_feats = np.zeros(
                (self.max_action_len, img_feats.shape[1]), dtype=np.float32
            )

            planner_img_feats[
                : self.episodes[idx].planner_action_length
            ] = img_feats[planner_pos_queue_idx]

            planner_actions_in = planner_actions.clone() - 1
            planner_actions_out = planner_actions[1:].clone() - 2

            planner_actions_in[planner_action_length:].fill_(0)
            planner_mask = planner_actions_out.clone().gt(-1)

            if len(planner_actions_out) > planner_action_length:
                planner_actions_out[planner_action_length:].fill_(0)

            controller_img_feats = np.zeros(
                (self.max_action_len, img_feats.shape[1]), dtype=np.float32
            )
            controller_img_feats[:controller_action_length] = img_feats[
                controller_pos_queue_idx
            ]

            controller_actions_in = self.episodes[idx].actions.clone() - 2

            if len(controller_actions_in) > controller_action_length:
                controller_actions_in[controller_action_length:].fill_(0)

            controller_out = controller_actions
            controller_mask = controller_out.clone().gt(-1)
            if len(controller_out) > controller_action_length:
                controller_out[controller_action_length:].fill_(0)

            # zero out forced controller return
            for i in range(controller_action_length):
                if (
                    i >= self.max_controller_actions - 1
                    and controller_out[i] == 0
                    and (
                        self.max_controller_actions == 1
                        or controller_out[
                            i - self.max_controller_actions + 1 : i  # noqa
                        ].sum()
                        == self.max_controller_actions - 1
                    )
                ):
                    controller_mask[i] = 0

            return (
                idx,
                question,
                answer,
                planner_img_feats,
                planner_actions_in,
                planner_actions_out,
                planner_action_length,
                planner_mask,
                controller_img_feats,
                controller_actions_in,
                planner_hidden_idx,
                controller_out,
                controller_action_length,
                controller_mask,
            )
