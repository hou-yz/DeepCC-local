from __future__ import print_function
import numpy as np
import h5py
import time
from collections import defaultdict
from torch.utils.data import Dataset


class HyperFeat(Dataset):
    def __init__(self, root, feature_dim=256):
        self.root = root
        h5file = h5py.File(self.root, 'r')
        self.data = np.array(h5file['hyperGT'])
        self.data = self.data[self.data[:, 1] != -1, :]  # rm -1 terms
        # iCam, pid, centerFrame, SpaGrpID, pos*2, v*2, 0, 256-dim feat
        self.feat_col = list(range(9, feature_dim + 9))
        self.motion_col = [0, 2, 4, 5, 6, 7]
        # train frame: [47720:187540]; val frame: [187541:227540]

        self.indexs = list(range(self.data.shape[0]))
        all_groupIDs = np.int_(np.unique(self.data[:, 3]))
        self.num_spatialGroup = len(all_groupIDs)
        self.min_groupID = min(all_groupIDs)
        self.pid_dic = defaultdict()
        self.index_by_pid_dic = defaultdict(list)
        self.index_by_SGid_pid_dic = defaultdict(dict)
        self.index_by_SGid_pid_icam_dic = defaultdict(dict)
        self.pid_by_SGid_dic = defaultdict(list)
        self.index_by_SGid_dic = defaultdict(list)
        for index in self.indexs:
            [icam, pid, spaGrpID] = self.data[index, [0, 1, 3]]
            icam, pid, spaGrpID = int(icam), int(pid), int(spaGrpID)

            if index not in self.index_by_pid_dic[pid]:
                self.index_by_pid_dic[pid].append(index)

            if spaGrpID not in self.index_by_SGid_pid_dic:
                self.index_by_SGid_pid_dic[spaGrpID] = defaultdict(list)
            self.index_by_SGid_pid_dic[spaGrpID][pid].append(index)

            if spaGrpID not in self.index_by_SGid_pid_icam_dic:
                self.index_by_SGid_pid_icam_dic[spaGrpID] = defaultdict(dict)
            if pid not in self.index_by_SGid_pid_icam_dic[spaGrpID]:
                self.index_by_SGid_pid_icam_dic[spaGrpID][pid] = defaultdict(list)
            self.index_by_SGid_pid_icam_dic[spaGrpID][pid][icam].append(index)

            if pid not in self.pid_by_SGid_dic[spaGrpID]:
                self.pid_by_SGid_dic[spaGrpID].append(pid)

            self.index_by_SGid_dic[spaGrpID].append(index)
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
    def __init__(self, h_dataset, train=True, L3=False, motion=False):
        self.h_dataset = h_dataset
        self.train = train
        self.num_spatialGroup = h_dataset.num_spatialGroup
        self.L3 = L3
        self.motion = motion

    def __len__(self):
        return len(self.h_dataset)

    def __getitem__(self, index):
        feat1, motion1, pid1, spaGrpID1 = self.h_dataset.__getitem__(index)
        cam1 = int(motion1[0])
        # if self.train:
        #     # 1:1 ratio for pos/neg
        #     target = np.random.randint(0, 2)
        # else:
        #     rand = np.random.rand()
        #     target = rand < 1 / len(self.h_dataset.pid_by_SGid_dic[spaGrpID1])
        target = np.random.randint(0, 2)
        if pid1 == -1:
            target = 0

        t0 = time.time()

        # if self.L3 and not self.motion:
        #     if len(self.h_dataset.index_by_SGid_pid_icam_dic[spaGrpID1][pid1]) > 1:
        #         target = np.random.rand() < 0.75

        # 1 for same
        if target == 1:
            index_pool = self.h_dataset.index_by_SGid_pid_dic[spaGrpID1][pid1]
            cam_pool = list(self.h_dataset.index_by_SGid_pid_icam_dic[spaGrpID1][pid1].keys())
            siamese_index = index
            if len(index_pool) > 1:
                cam2 = cam1
                if (self.L3 or self.motion) and len(cam_pool) > 1:
                    while cam2 == cam1:
                        cam2 = np.random.choice(cam_pool)
                    index_pool = self.h_dataset.index_by_SGid_pid_icam_dic[spaGrpID1][pid1][cam2]
                    siamese_index = np.random.choice(index_pool)
                else:
                    while siamese_index == index:
                        siamese_index = np.random.choice(index_pool)
        # 0 for different
        else:
            if not self.motion:
                pid_pool = list(self.h_dataset.index_by_SGid_pid_dic[spaGrpID1].keys())
            else:
                pid_pool = list(self.h_dataset.index_by_pid_dic.keys())
            pid2 = pid1
            if len(pid_pool) > 1:
                while pid2 == pid1:
                    pid2 = np.random.choice(pid_pool)
            if not self.motion:
                index_pool = self.h_dataset.index_by_SGid_pid_dic[spaGrpID1][pid2]
            else:
                index_pool = self.h_dataset.index_by_pid_dic[pid2]

            siamese_index = np.random.choice(index_pool)

        t1 = time.time()
        t_batch = t1 - t0
        feat2, motion2, pid2, spaGrpID2 = self.h_dataset.__getitem__(siamese_index)
        if target != (pid1 == pid2):
            target = (pid1 == pid2)
            pass
        if self.motion:
            # iCam, centerFrame, pos*2, v*2\
            if motion1[1] > motion2[1]:
                motion1, motion2 = motion2, motion1
            feat1, feat2 = np.insert(motion1[1:], [3, 3, 5, 5], [0, 0, 0, 0]), \
                           np.insert(motion2[1:], [1, 1, 3, 3], [0, 0, 0, 0])
            feat1 = np.insert(-feat1[1:], 0, feat1[0])
            data = (feat2 - feat1)
            data = np.concatenate((data[1:5], data[5:] * data[0]))
        else:
            data = abs(feat2 - feat1)

        return data, target
