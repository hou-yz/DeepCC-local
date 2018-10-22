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


class HyperFeat:

    # data = np.transpose(data)

    def __init__(self, root):
        self.root = os.path.expanduser(root)
        h5file = h5py.File(self.root, 'r')
        self.data = np.array(h5file['hyperGT'])

        self.indexs = list(range(self.data.shape[0]))
        # self.pid_hash = {}
        # all_pids = np.int_(np.unique(self.data[:, 1]))
        # for pid in all_pids:
        #     if pid not in self.pid_hash:
        #         self.pid_hash[pid] = len(self.pid_hash)
        self.groupID_hash = {}
        all_groupIDs = np.int_(np.unique(self.data[:, 3]))
        for groupID in all_groupIDs:
            if groupID not in self.groupID_hash:
                self.groupID_hash[groupID] = len(self.groupID_hash) + 1
        self.num_spatialGroup = len(self.groupID_hash)

        # self.hardGroups = []
        # self.spaGrpID_dic = [[] for _ in range(self.num_spatialGroup)]
        # self.pid_dic = [[] for _ in range(self.num_spatialGroup)]
        # self.download()
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
        spaGrpID = self.groupID_hash[int(self.data[index, 3])]
        return feat, pid, spaGrpID

    def __len__(self):
        return len(self.indexs)

    def download(self):
        for spatialGroupID in range(1, self.num_spatialGroup + 1):

            indices = np.nonzero(self.data[:, 3] == spatialGroupID)[0].tolist()
            for index in indices:
                # pid = self.pid_hash[np.int_(self.data[index, 1])]
                pid = int(self.data[index, 1])
                spaGrpID = int(self.data[index, 3])
                assert spaGrpID == spatialGroupID
                self.spaGrpID_dic[spaGrpID].append(index)
                self.pid_dic[spaGrpID].append(pid)
            if len(np.unique(self.spaGrpID_dic[spatialGroupID])) > 1:
                self.hardGroups.append(spatialGroupID)
        pass
