# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Tuple,
    Union,
)

import numpy as np
import torch
import webdataset as wds
import webdataset.filters as filters
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.core.simulator import ShortestPathPoint
from habitat.core.utils import try_cv2_import
from habitat.datasets.utils import VocabDict
from habitat_baselines.il.models.models import MultitaskCNN
from habitat_baselines.utils.common import (
    base_plus_ext,
    create_tar_archive,
    delete_folder,
    get_scene_episode_dict,
    valid_sample,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat.task.nav import NavigationEpisode

cv2 = try_cv2_import()


class NavDataset(wds.Dataset):
    """Pytorch dataset for PACMAN based navigation"""

    def __init__(
        self,
        config: "DictConfig",
        env: habitat.Env,
        device: torch.device,
        max_controller_actions: int = 5,
    ):
        """
        Args:
            config: DictConfig
            env: habitat Env
            device: torch.device
            max_controller_actions (int)
        """
        self.config = config.habitat
        self.env = env
        self.episodes: List[NavigationEpisode] = self.env._dataset.episodes
        self.max_controller_actions = max_controller_actions
        self.device = device
        self.sim = self.env.sim

        # sorting and making episode ids consecutive for simpler indexing
        self.sort_episodes()

        self.q_vocab = self.env._dataset.question_vocab  # type:ignore
        self.ans_vocab = self.env._dataset.answer_vocab  # type:ignore

        self.eval_save_results = config.habitat_baselines.eval_save_results

        if self.config.dataset.split == config.habitat_baselines.eval.split:
            self.mode = "val"
        else:
            self.mode = "train"

        self.frame_dataset_path = (
            config.habitat_baselines.frame_dataset_path.format(split=self.mode)
        )
        self.calc_max_length()
        self.restructure_ans_vocab()

        cnn_kwargs = {
            "only_encoder": True,
            "checkpoint_path": config.habitat_baselines.eqa_cnn_pretrain_ckpt_path,
        }
        self.cnn = MultitaskCNN(**cnn_kwargs)
        self.cnn.eval()
        self.cnn.to(self.device)

        self.scene_episode_dict = get_scene_episode_dict(self.episodes)
        self.preprocess_actions()
        if self.mode == "val":
            ctr = 0
            # ids in a way that episodes with same scenes are grouped together
            for scene in tqdm(
                self.scene_episode_dict.keys(),
                desc="going through all scenes from dataset",
            ):
                for episode in self.scene_episode_dict[scene]:
                    episode.episode_id = ctr
                    ctr += 1

        self.sort_episodes(consecutive_ids=False)

        group_by_keys = filters.Curried(self.group_by_keys_)
        super().__init__(
            urls=self.frame_dataset_path + ".tar",
            initial_pipeline=[group_by_keys()],
        )

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
            ctr = 0
            for scene in tqdm(
                list(self.scene_episode_dict.keys()),
                desc="Going through all scenes from dataset",
            ):
                self.load_scene(scene)

                for episode in tqdm(
                    self.scene_episode_dict[scene],
                    desc="Saving episode frames for each scene",
                ):
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

    def flat_to_hierarchical_actions(
        self, actions: Union[List[int], np.ndarray], controller_action_lim: int
    ):
        assert len(actions) != 0

        controller_action_ctr = 0

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

    def get_img_features(
        self, img: np.ndarray, preprocess: bool = False
    ) -> torch.Tensor:
        if preprocess:
            img_t = (
                (torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0)
                .view(1, 3, 256, 256)
                .to(self.device)
            )

        with torch.no_grad():
            return self.cnn(img_t)

    def get_hierarchical_features_till_spawn(
        self,
        idx: int,
        actions: np.ndarray,
        backtrack_steps: int = 0,
        max_controller_actions: int = 5,
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
        raw_img_feats = (
            self.get_img_features(self.frame_queue).cpu().numpy().copy()
        )

        controller_img_feat = torch.from_numpy(
            raw_img_feats[target_pos_idx].copy()
        )
        controller_action_in = pa_pruned[-1] - 2

        planner_img_feats = torch.from_numpy(
            raw_img_feats[pq_idx_pruned].copy()
        )
        planner_actions_in = torch.from_numpy(np.array(pa_pruned[:-1]) - 1)

        init_pos = self.episodes[idx].shortest_paths[0][target_pos_idx]

        return (
            planner_actions_in,
            planner_img_feats,
            controller_step,
            controller_action_in,
            controller_img_feat,
            init_pos,
            counter,
        )

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
            (
                planner_actions,
                controller_actions,
                pq_idx,
                cq_idx,
                ph_idx,
            ) = self.flat_to_hierarchical_actions(
                actions=ep.actions,
                controller_action_lim=self.max_controller_actions,
            )

            # padding actions with 0
            diff = self.max_action_len - ep.action_length
            for _ in range(diff):
                ep.actions.append(0)

            ep.actions = torch.Tensor(ep.actions)
            ep.planner_actions = ep.actions.clone().fill_(0)
            ep.controller_actions = ep.actions.clone().fill_(-1)

            ep.planner_hidden_idx = ep.actions.clone().fill_(0)
            ep.planner_pos_queue_idx, ep.controller_pos_queue_idx = [], []

            ep.planner_actions[: len(planner_actions)] = torch.Tensor(
                planner_actions
            )
            ep.controller_actions[: len(controller_actions)] = torch.Tensor(
                controller_actions
            )

            ep.planner_action_length = len(planner_actions) - 1
            ep.controller_action_length = len(controller_actions)

            ep.planner_pos_queue_idx.append(pq_idx)
            ep.controller_pos_queue_idx.append(cq_idx)

            ep.planner_hidden_idx[: len(controller_actions)] = torch.Tensor(
                ph_idx
            )

    def group_by_keys_(
        self,
        data: Generator,
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

                current_sample["question"] = np.array(question, dtype=np.int_)
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

    def sort_episodes(self, consecutive_ids: bool = True) -> None:
        # TODO: can be done in mp3d_eqa_dataset class too?
        self.episodes = sorted(self.episodes, key=lambda x: int(x.episode_id))
        if consecutive_ids:
            for idx, ep in enumerate(self.episodes):
                ep.episode_id = idx

    def save_frame_queue(
        self,
        pos_queue: List[ShortestPathPoint],
        episode_id: str,
    ) -> None:
        r"""Writes episode's frame queue to disk."""
        for idx, pos in enumerate(pos_queue):
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

    def cache_exists(self) -> bool:
        if os.path.exists(self.frame_dataset_path + ".tar"):
            return True
        else:
            os.makedirs(self.frame_dataset_path, exist_ok=True)
            return False

    def load_scene(self, scene: str) -> None:
        self.config.defrost()
        self.config.simulator.scene = scene
        self.config.freeze()
        self.env.sim.reconfigure(self.config.simulator)

    def map_dataset_sample(self, x: Dict) -> Tuple:
        """Mapper function to pre-process webdataset sample, example:
        img features, planner & controller actions etc.
        Args:
            x: webdataset sample containing ep_id, question, answer and imgs
        Returns:
            Processed sample containing img features, planner & controller actions etc.
        """
        idx = x["episode_id"]
        question = x["question"]
        answer = x["answer"]

        if len(question) < self.max_q_len:
            diff = self.max_q_len - len(question)
            for _ in range(diff):
                question.append(0)

        self.frame_queue = np.array(
            [img.transpose(2, 0, 1) / 255.0 for img in list(x.values())[4:]]
        )
        self.frame_queue = torch.Tensor(self.frame_queue).to(self.device)

        if self.mode == "val":
            # works only with batch size 1
            actions = self.episodes[idx].actions
            action_length = self.episodes[idx].action_length
            scene = self.episodes[idx].scene_id
            if scene != self.config.simulator.scene:
                logger.info("[ Loading scene - {}]".format(scene))
                self.config.defrost()
                self.config.simulator.scene = scene
                self.config.freeze()
                self.env.sim.reconfigure(self.config.simulator)

            goal_pos = self.episodes[idx].goals[0].position

            return idx, question, answer, actions, action_length, goal_pos

        planner_actions = self.episodes[idx].planner_actions
        controller_actions = self.episodes[idx].controller_actions

        planner_hidden_idx = self.episodes[idx].planner_hidden_idx

        planner_action_length = self.episodes[idx].planner_action_length
        controller_action_length = self.episodes[idx].controller_action_length

        raw_img_feats = (
            self.get_img_features(self.frame_queue).cpu().numpy().copy()
        )
        img_feats = np.zeros(
            (self.max_action_len, raw_img_feats.shape[1]), dtype=np.float32
        )
        img_feats[: raw_img_feats.shape[0], :] = raw_img_feats.copy()

        planner_pos_queue_idx = self.episodes[idx].planner_pos_queue_idx
        controller_pos_queue_idx = self.episodes[idx].controller_pos_queue_idx

        planner_img_feats = np.zeros(
            (self.max_action_len, img_feats.shape[1]), dtype=np.float32
        )

        planner_img_feats[
            : self.episodes[idx].planner_action_length
        ] = img_feats[tuple(planner_pos_queue_idx)]

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
            tuple(controller_pos_queue_idx)
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
                        i - self.max_controller_actions + 1 : i
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

    def __len__(self) -> int:
        return len(self.episodes)
