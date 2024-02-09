# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import List

import lmdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.core.simulator import ShortestPathPoint


class EQACNNPretrainDataset(Dataset):
    """Pytorch dataset for Embodied Q&A's feature-extractor"""

    def __init__(self, config, mode="train"):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        self.config = config.habitat
        self.dataset_path = config.habitat_baselines.dataset_path.format(
            split=mode
        )

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """
            self.env = habitat.Env(config=self.config)
            self.episodes = self.env._dataset.episodes

            logger.info(
                "Dataset cache not found. Saving rgb, seg, depth scene images"
            )
            logger.info(
                "Number of {} episodes: {}".format(mode, len(self.episodes))
            )

            self.scene_ids: List[str] = []
            self.scene_episode_dict = {}

            # dict for storing list of episodes for each scene
            for episode in self.episodes:
                if episode.scene_id not in self.scene_ids:
                    self.scene_ids.append(episode.scene_id)
                    self.scene_episode_dict[episode.scene_id] = [episode]
                else:
                    self.scene_episode_dict[episode.scene_id].append(episode)

            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )

            self.count = 0

            for scene in tqdm(list(self.scene_episode_dict.keys())):
                self.load_scene(scene)
                for episode in tqdm(self.scene_episode_dict[scene]):
                    try:
                        # TODO: Consider alternative for shortest_paths
                        pos_queue = episode.shortest_paths[0]  # type:ignore
                    except AttributeError as e:
                        logger.error(e)

                    random_pos = random.sample(pos_queue, 9)
                    self.save_frames(random_pos)

            logger.info("EQA-CNN-PRETRAIN database ready!")
            self.env.close()

        else:
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / 3)
        self.lmdb_env.close()
        self.lmdb_env = None

    def save_frames(self, pos_queue: List[ShortestPathPoint]) -> None:
        r"""
        Writes rgb, seg, depth frames to LMDB.
        """

        for pos in pos_queue:
            observation = self.env.sim.get_observations_at(
                pos.position, pos.rotation
            )

            depth = observation["depth"]
            rgb = observation["rgb"]

            scene = self.env.sim.semantic_annotations()  # type:ignore
            instance_id_to_label_id = {
                int(obj.id.split("_")[-1]): obj.category.index()
                for obj in scene.objects
            }
            self.mapping = np.array(
                [
                    instance_id_to_label_id[i]
                    for i in range(len(instance_id_to_label_id))
                ]
            )
            seg = np.take(self.mapping, observation["semantic"])
            seg[seg == -1] = 0
            seg = seg.astype("uint8")

            sample_key = "{0:0=6d}".format(self.count)
            with self.lmdb_env.begin(write=True) as txn:
                txn.put((sample_key + "_rgb").encode(), rgb.tobytes())
                txn.put((sample_key + "_depth").encode(), depth.tobytes())
                txn.put((sample_key + "_seg").encode(), seg.tobytes())

            self.count += 1

    def cache_exists(self) -> bool:
        if os.path.exists(self.dataset_path):
            if os.listdir(self.dataset_path):
                return True
        else:
            os.makedirs(self.dataset_path)
        return False

    def load_scene(self, scene) -> None:
        self.config.defrost()
        self.config.simulator.scene = scene
        self.config.freeze()
        self.env.sim.reconfigure(self.config.simulator)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()

        rgb_idx = "{0:0=6d}_rgb".format(idx)
        rgb_binary = self.lmdb_cursor.get(rgb_idx.encode())
        rgb_np = np.frombuffer(rgb_binary, dtype="uint8")
        rgb = rgb_np.reshape(256, 256, 3) / 255.0
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)

        depth_idx = "{0:0=6d}_depth".format(idx)
        depth_binary = self.lmdb_cursor.get(depth_idx.encode())
        depth_np = np.frombuffer(depth_binary, dtype="float32")
        depth = depth_np.reshape(1, 256, 256)

        seg_idx = "{0:0=6d}_seg".format(idx)
        seg_binary = self.lmdb_cursor.get(seg_idx.encode())
        seg_np = np.frombuffer(seg_binary, dtype="uint8")
        seg = seg_np.reshape(256, 256)

        return idx, rgb, depth, seg
