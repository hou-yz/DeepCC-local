from __future__ import print_function
import numpy as np
import h5py
import time
from collections import defaultdict
from torch.utils.data import Dataset


class bboxFeat(Dataset):
    def __init__(self, root, feature_dim=256, trainval='train', L='L2', window='75'):
        self.root = root
        h5file = h5py.File(self.root, 'r')
        self.L = L
        if window != 'Inf':
            self.window = int(window)
        else:
            self.window = np.inf

        self.data = np.array(h5file['emb'])
        # iCam, pid, centerFrame, 256-dim feat
        self.feat_col = list(range(3, 256 + 3))
        # train frame: [47720:187540]; val frame: [187541:227540]
        if trainval == 'train':
            self.frame_range = [47720, 187540]
        elif trainval == 'val':
            self.frame_range = [187541, 227540]
        else:
            self.frame_range = [47720, 227540]
        self.data = self.data[np.nonzero((self.data[:, 2] >= self.frame_range[0])
                                         & (self.data[:, 2] <= self.frame_range[1]))[0], :]
        self.GT_data = self.data[:, [0, 1, 2]]
        self.GT_data[:, 2] = (self.GT_data[:, 2] / self.window).astype(int)

        self.pid_dic = defaultdict()
        self.index_by_SGid_icam_pid_dic = defaultdict(dict)
        self.index_by_SGid_pid_dic = defaultdict(dict)
        self.index_by_SGid_pid_icam_dic = defaultdict(dict)
        self.pid_by_SGid_dic = defaultdict(list)
        self.index_by_SGid_dic = defaultdict(list)
        for index in range(self.GT_data.shape[0]):
            [icam, pid, spaGrpID] = self.GT_data[index, :]

            if spaGrpID not in self.index_by_SGid_icam_pid_dic:
                self.index_by_SGid_icam_pid_dic[spaGrpID] = defaultdict(dict)
            if icam not in self.index_by_SGid_icam_pid_dic[spaGrpID]:
                self.index_by_SGid_icam_pid_dic[spaGrpID][icam] = defaultdict(list)
            self.index_by_SGid_icam_pid_dic[spaGrpID][icam][pid].append(index)

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
        icam, pid, SGrp = map(int, self.GT_data[index, :])

        return feat, icam, pid, SGrp
        # target = np.random.randint(0, 2)
        #
        # t0 = time.time()
        # # window_range = [frame1 - int(self.window / 2), frame1 + int(self.window / 2)]
        # temporal_window_indexs = np.nonzero(self.GT_data[:, 2] == SGrp1)[0]
        # if 'L2' in self.L:
        #     same_cam_indexs = np.nonzero(self.GT_data[:, 0] == icam1)[0]
        #     temporal_window_indexs = np.intersect1d(temporal_window_indexs, same_cam_indexs)
        #
        # # 1 for same
        # if target == 1:
        #     index_pool = temporal_window_indexs[self.GT_data[temporal_window_indexs, 1] == pid1]
        # # 0 for different
        # else:
        #     index_pool = temporal_window_indexs[self.GT_data[temporal_window_indexs, 1] != pid1]
        #
        # if index_pool.size == 0:
        #     index_pool = temporal_window_indexs
        #     pass
        #
        # siamese_index = np.random.choice(index_pool)
        #
        # t1 = time.time()
        # t_batch = t1 - t0
        # feat2 = self.data[siamese_index, self.feat_col]
        # icam2, pid2, frame2 = map(int, self.GT_data[siamese_index, :])
        # if target != (pid1 == pid2):
        #     target = (pid1 == pid2)
        #     pass
        # data = abs(feat2 - feat1)
        #
        # return data, target

    def __len__(self):
        return self.GT_data.shape[0]


class SiamesebboxFeat(Dataset):
    def __init__(self, h_dataset):
        self.h_dataset = h_dataset

    def __len__(self):
        return len(self.h_dataset)

    def __getitem__(self, index):
        feat1, cam1, pid1, spaGrpID1 = self.h_dataset.__getitem__(index)
        target = np.random.randint(0, 2)

        t0 = time.time()

        # 1 for same
        if target == 1:
            cam_pool = list(self.h_dataset.index_by_SGid_pid_icam_dic[spaGrpID1][pid1].keys())
            siamese_index = index
            cam2 = cam1
            if ('L3' in self.h_dataset.L) and len(cam_pool) > 1:
                while cam2 == cam1:
                    cam2 = np.random.choice(cam_pool)
            index_pool = self.h_dataset.index_by_SGid_pid_icam_dic[spaGrpID1][pid1][cam2]

            if len(index_pool) > 1:
                while siamese_index == index:
                    siamese_index = np.random.choice(index_pool)
        # 0 for different
        else:
            cam_pool = list(self.h_dataset.index_by_SGid_icam_pid_dic[spaGrpID1].keys())
            cam2 = cam1
            if ('L3' in self.h_dataset.L) and len(cam_pool) > 1:
                while cam2 == cam1:
                    cam2 = np.random.choice(cam_pool)
            pid_pool = list(self.h_dataset.index_by_SGid_icam_pid_dic[spaGrpID1][cam2].keys())
            pid2 = pid1
            if len(pid_pool) > 1:
                while pid2 == pid1:
                    pid2 = np.random.choice(pid_pool)
            index_pool = self.h_dataset.index_by_SGid_icam_pid_dic[spaGrpID1][cam2][pid2]

            siamese_index = np.random.choice(index_pool)

        t1 = time.time()
        t_batch = t1 - t0
        feat2, cam2, pid2, spaGrpID2 = self.h_dataset.__getitem__(siamese_index)
        if target != (pid1 == pid2):
            target = (pid1 == pid2)
            pass
        data = abs(feat2 - feat1)

        return data, target
