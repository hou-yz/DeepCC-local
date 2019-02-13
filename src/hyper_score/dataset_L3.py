from __future__ import print_function
import numpy as np
import h5py
import time
from collections import defaultdict
from torch.utils.data import Dataset


class HyperFeat(Dataset):
    def __init__(self, root, feature_dim=256, motion_dim=9, trainval='train', L='L2', window='75'):
        self.root = root
        h5file = h5py.File(self.root, 'r')
        if motion_dim == 9:
            self.data = np.array(h5file['hyperGT'])
        else:
            self.data = np.array(h5file['emb'])
        self.data = self.data[self.data[:, 1] != -1, :]  # rm -1 terms
        self.L = L
        if window != 'Inf':
            window = int(window)
        else:
            window = np.inf
        if 'L3' in self.L:
            L2_window, L3_window = 300, window
        else:
            L2_window, L3_window = window, window

        # iCam, pid, centerFrame, SGrp2, pos*2, v*2, 0, 256-dim feat
        self.feat_col = list(range(motion_dim, feature_dim + motion_dim))
        self.motion_col = [0, 2, 4, 5, 6, 7]
        # train frame: [47720:187540]; val frame: [187541:227540]
        if trainval == 'train':
            self.frame_range = [47720, 187540]
        elif trainval == 'val':
            self.frame_range = [187541, 227540]
        else:
            self.frame_range = [47720, 227540]
        self.data = self.data[np.nonzero((self.data[:, 2] >= self.frame_range[0])
                                         & (self.data[:, 2] <= self.frame_range[1]))[0], :]
        self.GT_data = self.data[:, [0, 1, 2, 3]]
        self.GT_data[:, 2] = (self.data[:, 2] / L2_window).astype(int)
        self.GT_data[:, 3] = (self.data[:, 2] / L3_window).astype(int)

        self.pid_dic = defaultdict()
        self.index_by_SGrp2_icam_pid_dic = defaultdict(dict)
        self.index_by_SGrp2_pid_dic = defaultdict(dict)
        self.index_by_SGrp2_pid_icam_dic = defaultdict(dict)
        self.pid_by_SGrp2_dic = defaultdict(list)
        self.index_by_SGrp2_dic = defaultdict(list)
        self.index_by_SGrp3_icam_pid_dic = defaultdict(dict)
        self.index_by_SGrp3_pid_dic = defaultdict(dict)
        self.index_by_SGrp3_pid_icam_dic = defaultdict(dict)
        self.pid_by_SGrp3_dic = defaultdict(list)
        self.index_by_SGrp3_dic = defaultdict(list)
        for index in range(self.GT_data.shape[0]):
            [icam, pid, SGrp2, SGrp3] = self.GT_data[index, :]

            if SGrp2 not in self.index_by_SGrp2_icam_pid_dic:
                self.index_by_SGrp2_icam_pid_dic[SGrp2] = defaultdict(dict)
            if icam not in self.index_by_SGrp2_icam_pid_dic[SGrp2]:
                self.index_by_SGrp2_icam_pid_dic[SGrp2][icam] = defaultdict(list)
            self.index_by_SGrp2_icam_pid_dic[SGrp2][icam][pid].append(index)
            if SGrp2 not in self.index_by_SGrp2_pid_dic:
                self.index_by_SGrp2_pid_dic[SGrp2] = defaultdict(list)
            self.index_by_SGrp2_pid_dic[SGrp2][pid].append(index)
            if SGrp2 not in self.index_by_SGrp2_pid_icam_dic:
                self.index_by_SGrp2_pid_icam_dic[SGrp2] = defaultdict(dict)
            if pid not in self.index_by_SGrp2_pid_icam_dic[SGrp2]:
                self.index_by_SGrp2_pid_icam_dic[SGrp2][pid] = defaultdict(list)
            self.index_by_SGrp2_pid_icam_dic[SGrp2][pid][icam].append(index)
            if pid not in self.pid_by_SGrp2_dic[SGrp2]:
                self.pid_by_SGrp2_dic[SGrp2].append(pid)
            self.index_by_SGrp2_dic[SGrp2].append(index)

            if SGrp3 not in self.index_by_SGrp3_icam_pid_dic:
                self.index_by_SGrp3_icam_pid_dic[SGrp3] = defaultdict(dict)
            if icam not in self.index_by_SGrp3_icam_pid_dic[SGrp3]:
                self.index_by_SGrp3_icam_pid_dic[SGrp3][icam] = defaultdict(list)
            self.index_by_SGrp3_icam_pid_dic[SGrp3][icam][pid].append(index)
            if SGrp3 not in self.index_by_SGrp3_pid_dic:
                self.index_by_SGrp3_pid_dic[SGrp3] = defaultdict(list)
            self.index_by_SGrp3_pid_dic[SGrp3][pid].append(index)
            if SGrp3 not in self.index_by_SGrp3_pid_icam_dic:
                self.index_by_SGrp3_pid_icam_dic[SGrp3] = defaultdict(dict)
            if pid not in self.index_by_SGrp3_pid_icam_dic[SGrp3]:
                self.index_by_SGrp3_pid_icam_dic[SGrp3][pid] = defaultdict(list)
            self.index_by_SGrp3_pid_icam_dic[SGrp3][pid][icam].append(index)
            if pid not in self.pid_by_SGrp3_dic[SGrp3]:
                self.pid_by_SGrp3_dic[SGrp3].append(pid)
            self.index_by_SGrp3_dic[SGrp3].append(index)
        pass

    def __getitem__(self, index):
        feat = self.data[index, self.feat_col]
        motion = self.data[index, self.motion_col]
        _, pid, SGrp2, SGrp3 = map(int, self.GT_data[index, :])
        return feat, motion, pid, SGrp2, SGrp3

    def __len__(self):
        return self.GT_data.shape[0]


class SiameseHyperFeat(Dataset):
    def __init__(self, h_dataset, motion=False):
        self.h_dataset = h_dataset
        self.motion = motion

    def __len__(self):
        return len(self.h_dataset)

    def __getitem__(self, index):
        feat1, motion1, pid1, SGrp21, SGrp31 = self.h_dataset.__getitem__(index)
        cam1 = int(motion1[0])
        target = np.random.randint(0, 2)
        if pid1 == -1:
            target = 0

        t0 = time.time()

        # 1 for same
        if target == 1:
            siamese_index = index
            cam2 = cam1
            if 'L3' in self.h_dataset.L:
                cam_pool = list(map(int, self.h_dataset.index_by_SGrp3_pid_icam_dic[SGrp31][pid1].keys()))
                cam_pool.remove(cam1)
                if cam_pool:
                    cam2 = np.random.choice(cam_pool)
                    index_pool = self.h_dataset.index_by_SGrp3_pid_icam_dic[SGrp31][pid1][cam2]
                else:
                    index_pool = self.h_dataset.index_by_SGrp2_pid_icam_dic[SGrp21][pid1][cam2]
            else:
                index_pool = self.h_dataset.index_by_SGrp2_pid_icam_dic[SGrp21][pid1][cam2]

            if len(index_pool) > 1:
                while siamese_index == index:
                    siamese_index = np.random.choice(index_pool)
        # 0 for different
        else:
            cam2 = cam1
            if 'L3' in self.h_dataset.L:
                cam_pool = list(self.h_dataset.index_by_SGrp3_pid_icam_dic[SGrp31][pid1].keys())
                if cam_pool:
                    cam2 = np.random.choice(cam_pool)
                    pid_pool = list(self.h_dataset.index_by_SGrp3_icam_pid_dic[SGrp31][cam2].keys())
                else:
                    pid_pool = list(self.h_dataset.index_by_SGrp2_icam_pid_dic[SGrp21][cam2].keys())
            else:
                pid_pool = list(self.h_dataset.index_by_SGrp2_icam_pid_dic[SGrp21][cam2].keys())
            pid2 = np.random.choice(pid_pool)
            if len(pid_pool) > 1 and cam2 == cam1:
                while pid2 == pid1:
                    pid2 = np.random.choice(pid_pool)
            if 'L3' in self.h_dataset.L:
                index_pool = self.h_dataset.index_by_SGrp3_icam_pid_dic[SGrp31][cam2][pid2]
            else:
                index_pool = self.h_dataset.index_by_SGrp2_icam_pid_dic[SGrp21][cam2][pid2]

            siamese_index = np.random.choice(index_pool)

        t1 = time.time()
        t_batch = t1 - t0
        feat2, motion2, pid2, SGrp22, SGrp32 = self.h_dataset.__getitem__(siamese_index)
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
