import os
import random
from typing import List

import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import habitat
from habitat.core.simulator import ShortestPathPoint


class EDFEDataset(Dataset):
    """Pytorch dataset for Embodied Q&A's feature-extractor"""

    def __init__(self, env, config, mode="train"):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        self.env = env
        self.sim = self.env.sim
        self.config = config.TASK_CONFIG
        self.mode = mode

        self.episodes = self.get_all_episodes(config)

        self.dataset_path = config.DATASET_PATH

        self.scene_ids = []
        self.scene_episode_dict = {}

        # preparing a dict that stores list of episodes for each scene
        for ep in self.episodes:
            if ep.scene_id not in self.scene_ids:
                self.scene_ids.append(ep.scene_id)
                self.scene_episode_dict[ep.scene_id] = [ep.episode_id]
            else:
                self.scene_episode_dict[ep.scene_id].append(ep.episode_id)

        self.disk_cache_exists = self.check_cache_exists()

        if not self.disk_cache_exists:
            try:
                self.make_dataset_dirs()
            except FileExistsError:
                pass
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to each scene
            """
            print("Saving rgb, seg, depth data to database.")

            train_lmdb = lmdb.open(
                self.dataset_path.format(split="train"),
                map_size=int(1e11),
                writemap=True,
            )

            val_lmdb = lmdb.open(
                self.dataset_path.format(split="val"),
                map_size=int(0.5e11),
                writemap=True,
            )

            self.train_count = -1
            self.val_count = -1

            self.train_txn = train_lmdb.begin(write=True)
            self.val_txn = val_lmdb.begin(write=True)

            for scene in tqdm(list(self.scene_episode_dict.keys())):

                self.load_scene(scene)

                for ep_id in tqdm(self.scene_episode_dict[scene]):

                    episode = next(
                        ep for ep in self.episodes if ep.episode_id == ep_id
                    )

                    pos_queue = episode.shortest_paths[0]
                    random_pos = random.sample(pos_queue, 9)
                    self.save_frames(random_pos)

            print("EDFE database ready!")

            if self.mode == "train":
                self.lmdb_txn = self.train_txn
            elif self.mode == "val":
                self.lmdb_txn = self.val_txn
        else:
            lmdb_env = lmdb.open(self.dataset_path.format(split=self.mode))
            self.lmdb_txn = lmdb_env.begin()

        self.lmdb_cursor = self.lmdb_txn.cursor()
        self.env.close()

    def save_frames(self, pos_queue: List[ShortestPathPoint]) -> None:
        r"""Writes rgb, seg, depth frames to LMDB.
        """

        for pos in pos_queue:

            observation = self.env.sim.get_observations_at(
                pos.position, pos.rotation
            )

            depth = observation["depth"]
            rgb = observation["rgb"]

            scene = self.env.sim.semantic_annotations()
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

            if random.random() < 0.8:
                txn = self.train_txn
                self.train_count += 1
                count = self.train_count
            else:
                txn = self.val_txn
                self.val_count += 1
                count = self.val_count

            sample_key = "{0:0=6d}".format(count)

            txn.put((sample_key + "_rgb").encode(), rgb.tobytes())
            txn.put((sample_key + "_depth").encode(), depth.tobytes())
            txn.put((sample_key + "_seg").encode(), seg.tobytes())

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

    def get_all_episodes(self, config) -> List:
        r"""Fetches all episodes (train, val) from EQAMP3D dataset
        """

        train_episodes = self.env._dataset.episodes

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        self.env.close()
        self.env = habitat.Env(config.TASK_CONFIG)

        val_episodes = self.env._dataset.episodes
        for ep in val_episodes:
            # most val episodes share 'episode_id' with train episodes
            ep.episode_id = str(int(ep.episode_id) + 11796)
        all_episodes = train_episodes + val_episodes

        return all_episodes

    def make_dataset_dirs(self) -> None:
        for split in ["train", "val"]:
            os.makedirs(self.dataset_path.format(split=split))

    def check_cache_exists(self) -> bool:
        for split in ["train", "val"]:
            if not os.path.exists(self.dataset_path.format(split=split)):
                return False
        return True

    def load_scene(self, scene) -> None:
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.freeze()
        self.env.sim.reconfigure(self.config.SIMULATOR)

    def __len__(self) -> int:
        return int(self.lmdb_txn.stat()["entries"] / 3)

    def __getitem__(self, idx: int):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        rgb_idx = "{0:0=6d}_rgb".format(idx)
        rgb_binary = self.lmdb_cursor.get(rgb_idx.encode())
        rgb_np = np.frombuffer(rgb_binary, dtype="uint8")
        rgb = rgb_np.reshape(256, 256, 3) / 255.0
        rgb = rgb.transpose(2, 0, 1)

        depth_idx = "{0:0=6d}_depth".format(idx)
        depth_binary = self.lmdb_cursor.get(depth_idx.encode())
        depth_np = np.frombuffer(depth_binary, dtype="float32")
        depth = depth_np.reshape(1, 256, 256)

        seg_idx = "{0:0=6d}_seg".format(idx)
        seg_binary = self.lmdb_cursor.get(seg_idx.encode())
        seg_np = np.frombuffer(seg_binary, dtype="uint8")
        seg = seg_np.reshape(256, 256)

        return idx, rgb, depth, seg
