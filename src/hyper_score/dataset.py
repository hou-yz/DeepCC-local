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
    def __init__(self, root, L2_speed):
        self.root = root
        h5file = h5py.File(self.root, 'r')
        self.data = np.array(h5file['hyperGT'])
        if L2_speed == 'mid':
            self.feat_col = [0, 2] + list(range(4, 8)) + list(range(9, 265))  # cam,frame,pos_x,pos_y,v_x,v_y,256-dim
        else:
            self.feat_col = [0, 2] + list(range(4, 14)) + list(range(15, 271))

        self.indexs = list(range(self.data.shape[0]))
        all_groupIDs = np.int_(np.unique(self.data[:, 3]))
        self.num_spatialGroup = len(all_groupIDs)
        self.min_groupID = min(all_groupIDs)
        self.index_pool_dic = defaultdict(dict)
        self.pid_pool_dic = defaultdict(list)
        for index in self.indexs:
            [pid, spaGrpID] = self.data[index, [1, 3]]
            pid, spaGrpID = int(pid), int(spaGrpID)
            if spaGrpID not in self.index_pool_dic:
                self.index_pool_dic[spaGrpID] = defaultdict(list)
            self.index_pool_dic[spaGrpID][pid].append(index)
            if pid not in self.pid_pool_dic[spaGrpID]:
                self.pid_pool_dic[spaGrpID].append(pid)
        pass

    def __getitem__(self, index):
        feat = self.data[index, self.feat_col]
        # pid = self.pid_hash[np.int_(self.data[index, 1])]
        pid = int(self.data[index, 1])
        spaGrpID = int(self.data[index, 3])
        return feat, pid, spaGrpID

    def __len__(self):
        return len(self.indexs)


class SiameseHyperFeat(Dataset):
    def __init__(self, h_dataset):
        self.h_dataset = h_dataset
        self.num_spatialGroup = h_dataset.num_spatialGroup

    def __len__(self):
        return len(self.h_dataset)

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        feat1, pid1, spaGrpID1 = self.h_dataset.__getitem__(index)
        if pid1 == -1:
            target = 0
        if target == 1:  # 1 for same
            index_pool = self.h_dataset.index_pool_dic[spaGrpID1][pid1]
            if len(index_pool) > 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(index_pool)
            else:
                siamese_index = np.random.choice(index_pool)
        else:  # 0 for different
            spatialGroupID = spaGrpID1
            pid_pool = self.h_dataset.pid_pool_dic[spatialGroupID]
            while len(pid_pool) <= 1:
                spatialGroupID += 1
                pid_pool = self.h_dataset.pid_pool_dic[spatialGroupID]
            siamese_pid = pid1
            while siamese_pid == pid1:
                siamese_pid = np.random.choice(pid_pool)
            index_pool = self.h_dataset.index_pool_dic[spatialGroupID][siamese_pid]
            if len(index_pool) > 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(index_pool)
            else:
                siamese_index = np.random.choice(index_pool)
        feat2, pid2, spaGrpID2 = self.h_dataset.__getitem__(siamese_index)
        if target != (pid1 == pid2):
            target = (pid1 == pid2)
            pass
        if feat1[1] < feat2[1]:
            return feat1, feat2, target
        else:
            return feat2, feat1, target
