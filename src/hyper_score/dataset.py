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
    def __init__(self, root):
        self.root = root
        h5file = h5py.File(self.root, 'r')
        self.data = np.array(h5file['hyperGT'])
        self.data = self.data[self.data[:, 1] != -1, :]
        # iCam, pid, centerFrame, SpaGrpID, pos*2, v*2, 0, 256-dim feat
        self.feat_col = list(range(9, 265))
        self.motion_col = [0, 2, 4, 5, 6, 7]
        # train frame: [47720:187540]; val frame: [187541:227540]

        self.indexs = list(range(self.data.shape[0]))
        all_groupIDs = np.int_(np.unique(self.data[:, 3]))
        self.num_spatialGroup = len(all_groupIDs)
        self.min_groupID = min(all_groupIDs)
        self.index_by_SGid_pid_dic = defaultdict(dict)
        self.pid_by_SGid_dic = defaultdict(list)
        self.index_by_icam_pid_dic = defaultdict(dict)
        for index in self.indexs:
            [icam,pid, spaGrpID] = self.data[index, [0,1, 3]]
            icam,pid, spaGrpID = int(icam),int(pid), int(spaGrpID)
            if spaGrpID not in self.index_by_SGid_pid_dic:
                self.index_by_SGid_pid_dic[spaGrpID] = defaultdict(list)
            self.index_by_SGid_pid_dic[spaGrpID][pid].append(index)
            if pid not in self.pid_by_SGid_dic[spaGrpID]:
                self.pid_by_SGid_dic[spaGrpID].append(pid)
            if icam not in self.index_by_icam_pid_dic:
                self.index_by_icam_pid_dic[icam] = defaultdict(list)
            self.index_by_icam_pid_dic[icam][pid].append(index)
        pass

    def __getitem__(self, index):
        feat = self.data[index, self.feat_col]
        motion = self.data[index, self.motion_col]
        # pid = self.pid_hash[np.int_(self.data[index, 1])]
        pid = int(self.data[index, 1])
        spaGrpID = int(self.data[index, 3])
        return feat, motion, pid, spaGrpID

    def __len__(self):
        return len(self.indexs)


class SiameseHyperFeat(Dataset):
    def __init__(self, h_dataset, train=True):
        self.h_dataset = h_dataset
        self.train = train
        self.num_spatialGroup = h_dataset.num_spatialGroup

    def __len__(self):
        return len(self.h_dataset)

    def __getitem__(self, index):
        feat1, motion1, pid1, spaGrpID1 = self.h_dataset.__getitem__(index)
        if self.train:
            # 1:1 ratio for pos/neg
            target = np.random.randint(0, 2)
        else:
            rand = np.random.rand()
            target = rand < 1 / len(self.h_dataset.pid_by_SGid_dic[spaGrpID1])
        if pid1 == -1:
            target = 0
        # 1 for same
        if target == 1:  
            # index_pool = self.h_dataset.index_by_icam_pid_dic[motion1[0]][pid1]
            index_pool = self.h_dataset.index_by_SGid_pid_dic[spaGrpID1][pid1]
            if len(index_pool) > 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(index_pool)
            else:
                siamese_index = np.random.choice(index_pool)
        # 0 for different
        else: 
            spatialGroupID = spaGrpID1
            pid_pool = self.h_dataset.pid_by_SGid_dic[spatialGroupID]
            while len(pid_pool) <= 1:
                spatialGroupID += 1
                pid_pool = self.h_dataset.pid_by_SGid_dic[spatialGroupID]
            siamese_pid = pid1
            while siamese_pid == pid1:
                siamese_pid = np.random.choice(pid_pool)
            index_pool = self.h_dataset.index_by_SGid_pid_dic[spatialGroupID][siamese_pid]
            if len(index_pool) > 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(index_pool)
            else:
                siamese_index = np.random.choice(index_pool)
        feat2, motion2, pid2, spaGrpID2 = self.h_dataset.__getitem__(siamese_index)
        if target != (pid1 == pid2):
            target = (pid1 == pid2)
            pass
        # if feat1[1] < feat2[1]:
        #     return feat1, feat2, target
        # else:
        #     return feat2, feat1, target
        if motion1[0] != motion2[0]:
            motion_score = 0
        else:
            frame_dif = motion2[1] - motion1[1]
            pos_dif = motion2[[2, 3]] - motion1[[2, 3]]
            forward_err = motion1[[4, 5]] * frame_dif - pos_dif
            backward_err = motion2[[4, 5]] * frame_dif - pos_dif
            error = min(np.linalg.norm(forward_err), np.linalg.norm(backward_err))
            motion_score = error / 2203  # norm for [1920,1080]
        return feat2, feat1, motion_score, target
