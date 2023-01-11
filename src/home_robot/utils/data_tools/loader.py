import glob
import h5py
import numpy as np
import os
import torch

import data_tools.base as base
import data_tools.image as image


class Trial(object):
    def __init__(self, name, h5_filename, dataset, group):
        """
        Use group for initialization
        """
        self.dataset = dataset
        self.name = name
        self.h5_filename = h5_filename
        self.group = None

        temporal_keys = group[base.TEMPORAL_KEYS][()]
        config_keys = group[base.CONFIG_KEYS][()]
        image_keys = group[base.IMAGE_KEYS][()]
        temporal_keys = (
            str(temporal_keys, "utf-8")
            if type(temporal_keys) == bytes
            else temporal_keys
        )
        config_keys = (
            str(config_keys, "utf-8") if type(config_keys) == bytes else config_keys
        )
        image_keys = (
            str(image_keys, "utf-8") if type(image_keys) == bytes else image_keys
        )
        self.temporal_keys = temporal_keys.split(",")
        self.config_keys = config_keys.split(",")
        self.image_keys = image_keys.split(",")
        if len(self.temporal_keys) > 0:
            self.length = group[self.temporal_keys[0]].shape[0]
        else:
            self.length = 0

    def __getitem__(self, key):
        if self.group is None:
            h5 = self.dataset.get_h5_file(self.h5_filename)
            self.group = h5[self.name]
        return self.group[key]

    def get_conf(self, key):
        assert key in self.config_keys
        group = self[key]
        return group[()]

    def get_dict(self, key):
        # get config dictionary
        group = self[key]
        conf = {}
        for k in group.keys():
            conf[k] = group[k][()]
        return conf

    def get_img(self, key, idx, depth=False, rgb=False, depth_factor=10000):
        assert key in self.image_keys
        group = self[key]
        arr = image.img_from_bytes(group[str(idx)][()])
        if depth:
            return arr / depth_factor
        elif rgb:
            return arr / 255.0
        else:
            return arr


class DatasetBase(torch.utils.data.Dataset):
    """Access hdf5 file(s) and creates data slices that we can use for training neural
    net models."""

    def __init__(
        self,
        dirname,
        template="*.h5",
        verbose=False,
        trial_list: list = None,
        TrialType=None,
    ):
        """
        Take all files in directory
        """
        if TrialType is None:
            self.Trial = Trial
        else:
            self.Trial = TrialType
        self.dirname = dirname
        self.template = template
        self.verbose = verbose
        self.trial_list = trial_list
        template = os.path.join(self.dirname, self.template)
        files = sorted(glob.glob(template))
        self.process_files(files)

    def get_h5_file(self, filename):
        if filename in self.h5s:
            return self.h5s[filename]
        else:
            h5 = h5py.File(filename, "r")
            self.h5s[filename] = h5
            return h5

    def process_files(self, files):
        """Read through the set of files and track unique files and everything else."""
        if self.verbose:
            print("Found these files:", files)
        self.trials = []
        self.h5s = {}
        lens = []
        for filename in files:
            # Check each file to see how many entires it has
            with h5py.File(filename, "r") as h5:
                for key, h5_trial in h5.items():
                    if not self.trial_list or (
                        self.trial_list and key in self.trial_list
                    ):
                        # Open the trial and extract metadata
                        trial = self.Trial(key, filename, self, h5_trial)
                        if self.verbose:
                            print("trial =", key, trial.length)
                        lens.append(trial.length)
                        # Bookkeeping for all the trials
                        self.trials.append(trial)

        self.trial_lengths = np.cumsum(lens)
        self.max_idx = self.trial_lengths[-1]

    def __len__(self):
        return self.max_idx

    def get_datum(self, trial, idx):
        raise NotImplementedError()

    def __getitem__(self, idx):
        trial_idx = np.argmax(idx < self.trial_lengths)
        trial = self.trials[trial_idx]
        if trial_idx > 0:
            idx = idx - self.trial_lengths[trial_idx - 1]
        return self.get_datum(trial, idx)
