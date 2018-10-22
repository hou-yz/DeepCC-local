from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path as osp
import errno
import numpy as np
import torch
import codecs
import h5py
from collections import defaultdict
from torch.utils.data import Dataset


class HyperFeat(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        h5file = h5py.File(self.root, 'r')
        self.data = np.array(h5file['hyperGT'])

        self.indexs = list(range(self.data.shape[0]))
        all_groupIDs = np.int_(np.unique(self.data[:, 3]))
        self.num_spatialGroup = len(all_groupIDs)
        self.min_groupID = min(all_groupIDs)
        self.spaGrpID_dic = defaultdict(list)
        self.pid_dic = defaultdict(list)
        for index in self.indexs:
            [pid, spaGrpID] = self.data[index, [1, 3]]
            self.spaGrpID_dic[int(spaGrpID)].append(index)
            self.pid_dic[int(spaGrpID)].append(int(pid))
        pass

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        feat_col = [0, 2] + list(range(4, 8)) + list(range(9, 265))  # cam,frame,pos_x,pos_y,v_x,v_y,256-dim
        feat = self.data[index, feat_col]
        # pid = self.pid_hash[np.int_(self.data[index, 1])]
        pid = int(self.data[index, 1])
        spaGrpID = int(self.data[index, 3])
        return feat, pid, spaGrpID

    def __len__(self):
        return len(self.indexs)


class SiameseHyperFeat(Dataset):
    def __init__(self, h_dataset):
        self.h_dataset = h_dataset

