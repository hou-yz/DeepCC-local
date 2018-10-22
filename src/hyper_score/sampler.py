from __future__ import absolute_import

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.spaGrpID_dic = data_source.spaGrpID_dic
        self.pid_dic = data_source.pid_dic
        self.spaGrpID_max = data_source.num_spatialGroup
        self.spaGrpID = 1

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        ret = []
        t_s = []
        for pid in np.unique(self.pid_dic[self.spaGrpID]):
            t = np.array(self.spaGrpID_dic[self.spaGrpID])[self.pid_dic[self.spaGrpID] == pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False).tolist()
            else:
                t = t.tolist()
            #     t = np.random.choice(t, size=self.num_instances, replace=True).tolist()
            t_s += t
        self.spaGrpID += 1
        if self.spaGrpID == self.spaGrpID_max + 1:
            self.spaGrpID = 1
        ret.extend(t_s)
        ret = np.unique(ret)
        return iter(ret)
