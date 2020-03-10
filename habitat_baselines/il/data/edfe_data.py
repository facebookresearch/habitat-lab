
import os
import random
from tqdm import tqdm
from typing import List

import cv2
import numpy as np
# import torch
from torch.utils.data import Dataset

import habitat
from habitat.core.simulator import ShortestPathPoint


class EDFEDataset(Dataset):
    """Pytorch dataset for Embodied Q&A's feature-extractor"""

    def __init__(
        self, env, config, mode='train'
    ):
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
            print('Saving rgb, semantic, depth frames to disk.')

            for scene in tqdm(list(self.scene_episode_dict.keys())):

                self.load_scene(scene)

                for ep_id in tqdm(self.scene_episode_dict[scene]):

                    episode = next(
                        ep for ep in self.episodes if ep.episode_id == ep_id
                    )

                    pos_queue = episode.shortest_paths[0]
                    random_pos = random.sample(pos_queue, 9)
                    self.save_frames(random_pos, ep_id)

            print('Saved all episodes\' frames to disk. Frame dataset ready.')
        self.env.close()

        self.rgb_list = sorted(os.listdir(self.dataset_path.format(
            split=self.mode, type='rgb'
        )))
        self.depth_list = sorted(os.listdir(self.dataset_path.format(
            split=self.mode, type='depth'
        )))
        self.semantic_list = sorted(os.listdir(self.dataset_path.format(
            split=self.mode, type='semantic'
        )))

    def save_frames(
        self, pos_queue: List[ShortestPathPoint], episode_id
    ) -> None:
        r"""Writes rgb, semantic, depth frames to disk.
        """

        for idx, pos in enumerate(pos_queue):

            observation = self.env.sim.get_observations_at(
                pos.position, pos.rotation
            )

            depth = observation["depth"]
            rgb = observation["rgb"]
            semantic = observation["semantic"]
            semantic = semantic % 40
            semantic = semantic.astype(np.uint8)

            if random.random() < 0.8:
                split = "train"
            else:
                split = "val"

            rgb_frame_path = os.path.join(self.dataset_path
                                          .format(split=split, type='rgb'),
                                          "ep_" + episode_id + "_{0:0=3d}"
                                          .format(idx))
            depth_frame_path = os.path.join(self.dataset_path
                                            .format(split=split, type='depth'),
                                            "ep_" + episode_id + "_{0:0=3d}"
                                            .format(idx))
            semantic_frame_path = os.path.join(self.dataset_path
                                               .format(split=split,
                                                       type='semantic'),
                                               "ep_" + episode_id + "_{0:0=3d}"
                                               .format(idx))

            cv2.imwrite(rgb_frame_path + '.jpg', rgb)
            cv2.imwrite(depth_frame_path + '.jpg', depth * 255)
            np.savez_compressed(semantic_frame_path, semantic)

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
            for type in ["rgb", "semantic", "depth"]:
                os.makedirs(self.dataset_path.format(split=split, type=type))

    def check_cache_exists(self) -> bool:
        for split in ["train", "val"]:
            for type in ["rgb", "semantic", "depth"]:
                if not os.path.exists(self.dataset_path.format(split=split,
                                                               type=type)):
                    return False
                else:
                    if len(os.listdir(
                           self.dataset_path
                           .format(split=split,
                                   type=type))) == 0:
                        return False
        return True

    def load_scene(self, scene) -> None:
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.freeze()
        self.env.sim.reconfigure(self.config.SIMULATOR)

    def __len__(self) -> int:
        return len(self.rgb_list)

    def __getitem__(
        self, idx: int
    ):
        r"""Returns batches to trainer.

        batch: (rgb, depth, semantic)

        """
        rgb = cv2.imread(os.path.join(
            self.dataset_path.format(split=self.mode, type='rgb'),
            self.rgb_list[idx]
        ))
        rgb = rgb.transpose(2, 0, 1)
        rgb = rgb / 255.0

        depth = cv2.imread(os.path.join(
            self.dataset_path.format(split=self.mode, type='depth'),
            self.depth_list[idx]
        ), -1)
        depth = depth / 255.0

        semantic = np.load(os.path.join(
            self.dataset_path.format(split=self.mode, type='semantic'),
            self.semantic_list[idx]
        ))['arr_0']

        return rgb, depth, semantic
