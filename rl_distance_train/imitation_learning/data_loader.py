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
        video_paths = glob.glob(f"{self.datapath}/[0-9]*")
        num_vid = len(video_paths)

        video_id = np.random.randint(0, int(num_vid)) 
        vid = f"{video_paths[video_id]}"

        loaded_video, _, _ = torchvision.io.read_video(os.path.join(vid, "trajectory.mp4"), pts_unit='sec', output_format='TCHW')
        loaded_video = loaded_video.to(torch.float32)
        vidlen = len(loaded_video)

        with open(os.path.join(vid, "actions.pkl"), 'rb') as f:
            actions = torch.tensor(pickle.load(f))
        assert len(actions) == vidlen
        
        traj = []
        selected_actions = []
        while len(selected_actions) < self.max_len:
            if self.deterministic:
                start_ind = max(0, vidlen - self.max_len)
            else:
                start_ind = random.randint(0, vidlen - 1)
        
            traj.extend(loaded_video[start_ind:vidlen])
            selected_actions.extend(actions[start_ind:vidlen])

        traj = torch.stack(traj[:self.max_len])
        selected_actions = torch.stack(selected_actions[:self.max_len])
        goal = loaded_video[-1]

        if self.doaug:
            traj = self.aug(traj / 255.0) * 255.0
            goal = self.aug(goal / 255.0) * 255.0            

        return (traj, goal, selected_actions)

    def __iter__(self):
        while True:
            yield self._sample()