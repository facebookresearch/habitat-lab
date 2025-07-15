import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from PIL import Image

import math
import cv2
import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random
import io
import zipfile
import av

## Data Loader
class DataBuffer(IterableDataset):
    def __init__(self, datapath, doaug=True, normalize_img=True, deterministic=False, max_len=250):
        """
        negative_mining (float): 0 no negative mining, 1 every example negative mining
        """
        self.datapath = datapath
        assert(self.datapath is not None)
        self.doaug = doaug
        self.max_len = max_len
        self.deterministic = deterministic
    
        if self.doaug:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            )
        else:
            self.aug = lambda a : a

    def _sample(self):        
        # Sample a video from datasource
        video_paths = glob.glob(f"{self.datapath}/[0-9]*.zip")
        num_vid = len(video_paths)

        video_id = np.random.randint(0, int(num_vid)) 
        vid = f"{video_paths[video_id]}"

        loaded_video, actions, gps_compass = self._read_from_zip(vid)
        loaded_video = loaded_video.to(torch.float32)
        vidlen = len(loaded_video)

        gps_list = gps_compass['gps']
        compass_list = gps_compass['compass']

        assert len(actions) == vidlen
        assert len(actions) == len(gps_list)
        assert len(actions) == len(compass_list)
        
        traj = []
        episode_gps = []
        episode_compass = []
        episode_actions = []
        while len(episode_actions) < self.max_len:
            if self.deterministic:
                start_ind = max(0, vidlen - self.max_len)
            else:
                start_ind = random.randint(0, vidlen - 1)
        
            traj.extend(loaded_video[start_ind:vidlen])
            episode_gps.extend(gps_list[start_ind:vidlen])
            episode_compass.extend(compass_list[start_ind:vidlen])
            episode_actions.extend(actions[start_ind:vidlen])

        traj = torch.stack(traj[:self.max_len])
        episode_gps = torch.stack([torch.from_numpy(arr) for arr in episode_gps[:self.max_len]])
        episode_compass = torch.stack([torch.from_numpy(arr) for arr in episode_compass[:self.max_len]])
        episode_actions = torch.tensor(episode_actions[:self.max_len])
        goal = loaded_video[-1]

        if self.doaug:
            traj = self.aug(traj / 255.0) * 255.0
            goal = self.aug(goal / 255.0) * 255.0            

        return (traj, goal, episode_gps, episode_compass, episode_actions)

    def __iter__(self):
        while True:
            yield self._sample()

    @staticmethod
    def _read_from_zip(zip_path: str):
        """
        Load 'trajectory.mp4', 'actions.pkl', and 'gps_compass.pkl' from the zip.
        Returns: (video_tensor, actions, gps_compass)
        - video_tensor: [T, C, H, W] float32
        - actions: whatever is stored in actions.pkl
        - gps_compass: dict with keys 'gps' and 'compass'
        """
        with zipfile.ZipFile(zip_path) as zf:
            mp4_bytes = zf.read("trajectory.mp4")
            actions_bytes = zf.read("actions.pkl")
            gps_compass_bytes = zf.read("gps_compass.pkl")
        
        # Decode video
        with io.BytesIO(mp4_bytes) as f:
            with av.open(f) as container:
                frames = [
                    torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
                    for frame in container.decode(video=0)
                ]
        if not frames:
            raise RuntimeError("No frames found in video")
        video_tensor = torch.stack(frames).float()  # [T, C, H, W]

        # Unpickle actions
        actions = pickle.loads(actions_bytes)

        # Unpickle gps_compass dict
        gps_compass = pickle.loads(gps_compass_bytes)  # should be a dict with keys 'gps' and 'compass'

        return video_tensor, actions, gps_compass