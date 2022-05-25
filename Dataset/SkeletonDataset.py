import numpy as np
import scipy.stats as ss
import torch

from torch.utils.data import Dataset
from typing import List

'''
Default Constants for Data Augmentation computed in previous analysis:
_POINTS_LOSS_PROB = Single loss point probability for each point
_DIST_PARAMS = Parameters of the distribution for the number of lost points
_LOSS_FORMAT = Notation for missing points
'''

_POINTS_LOSS_PROB = [0.15140283,
                     0.01488367,
                     0.03455748,
                     0.12043796,
                     0.15026232,
                     0.03159215,
                     0.11610401,
                     0.16474681,
                     0.04738823,
                     0.0854813,
                     0.13332573,
                     0.04972628,
                     0.08998631,
                     0.13383896,
                     0.22793111,
                     0.22234261,
                     0.27115648,
                     0.29071624]
_DIST_PARAMS = [0.0, 2.3358804744525545]
_LOSS_FORMAT = [-1, -1, -1]


class SkeletonDataset(Dataset):
    def __init__(self, skeletons: List, augment=True, dist_param=None,
                 points_loss_prob=None, loss_format=None):
        """
        :param skeletons: List of skeletons sequences, each of them with the shape [n_frames, n_points, dims]. Example:
        with Intel RealSense the dimension is [n_frames, 18, 3]
        :param augment: If True apply data augmentation
        :param dist_param: Exponential dist parameters [loc, scale]
        :param points_loss_prob: List of single loss point probability for each point
        :param loss_format: Notation for missing points [x, y, z]
        """
        self.examples = skeletons
        self.augment = augment
        if points_loss_prob is not None:
            self.points_loss_prob = points_loss_prob
        else:
            self.points_loss_prob = _POINTS_LOSS_PROB
        # Normalize
        self.points_loss_prob = [p/sum(self.points_loss_prob) for p in self.points_loss_prob]
        if dist_param is not None:
            self.dist_param = dist_param
        else:
            self.dist_param = _DIST_PARAMS
        if loss_format is not None:
            self.loss_format = loss_format
        else:
            self.loss_format = _LOSS_FORMAT

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.augment:
            masked_frames = self.simulate_loss(self.examples[i])
            return torch.tensor(masked_frames, dtype=torch.float32), \
                   torch.tensor(self.examples[i][1:], dtype=torch.float32)
        else:
            return torch.tensor(self.examples[i][0:-1], dtype=torch.float32), \
                   torch.tensor(self.examples[i][1:], dtype=torch.float32)

    def simulate_loss(self, example):
        """
        :param example: skeletons sequence with shape [n_frames, n_points, dims]
        :return: augmented skeleton sequence
        """
        frames = np.empty_like(example)
        frames[:, :, :] = example
        # First frame is unmasked
        masked_frames = [frames[0, :, :]]
        # Skip last frame that is only for prediction
        for frame in frames[1:-1, :, :]:
            # Exponential loss with prob
            n_loss = int(ss.expon.rvs(*self.dist_param))
            # Truncate the distribution at len(self.points_loss_prob)
            if n_loss > len(self.points_loss_prob):
                n_loss = len(self.points_loss_prob)
            # Select n_loss points each with his probability
            loss_points = np.random.choice(len(self.points_loss_prob), size=n_loss, replace=False,
                                           p=self.points_loss_prob)
            frame[loss_points, :] = self.loss_format
            masked_frames.append(frame)
        masked_frames = np.array(masked_frames)
        return masked_frames


class SkeletonDataset_Test(Dataset):
    def __init__(self, skeletons: List, labels: List):
        """
        :param skeletons: List of skeletons sequences, each of them with the shape [n_frames, n_points, dims]. Example:
        with Intel RealSense the dimension is [n_frames, 18, 3]
        :param labels: List of skeletons sequences, each of them with the shape [n_frames, n_points, dims]. Example:
        with Intel RealSense the dimension is [n_frames, 18, 3]
        """

        self.examples = skeletons
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i][0:-1], dtype=torch.float32), \
               torch.tensor(self.labels[i][1:], dtype=torch.float32)
