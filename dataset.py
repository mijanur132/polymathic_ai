import h5py
import numpy as np
import torch
import os
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, path, load_into_memory=True, shuffle=True):
        self.path = path
        self.data = h5py.File(path, "r")
        self.load_into_memory = load_into_memory
        if load_into_memory:
            inputs = np.stack([self.data["density"], self.data["recording_date"]], axis=1)
            labels = self.data["labels"][:]
            # Shuffle samples to randomize order, improving class mixing
            if shuffle:
                indices = np.arange(labels.shape[0])
                np.random.shuffle(indices)
                self.inputs = inputs[indices]
                self.labels = labels[indices]
            else:
                self.inputs = inputs
                self.labels = labels
            # Compute feature-wise mean and std for normalization
            self.means = self.inputs.astype(np.float32).mean(axis=(0,2,3))
            self.stds = self.inputs.astype(np.float32).std(axis=(0,2,3))

    def __len__(self):
        if self.load_into_memory:
            return self.inputs.shape[0]
        else:
            return self.data["density"].shape[0]

    def __getitem__(self, idx):
        if self.load_into_memory:
            # Normalize each sample using precomputed means and stds
            x = (self.inputs[idx].astype(np.float32) - self.means[:, None, None]) / self.stds[:, None, None]
            y = np.float32(self.labels[idx])
            return torch.from_numpy(x), torch.tensor(y)
        else:
            # Return sample as is (no normalization) if not loaded in memory
            x = np.stack([self.data["density"][idx], self.data["recording_date"][idx]], axis=1)
            return np.float32(x), np.float32(self.data["labels"][idx])
