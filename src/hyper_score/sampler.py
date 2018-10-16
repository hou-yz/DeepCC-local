from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.spaGrpID_dic = defaultdict(list)
        self.pid_dic = defaultdict(list)
        for index, (feat, pid, spaGrpID) in enumerate(data_source):
            self.spaGrpID_dic[spaGrpID].append(index)
            self.pid_dic[spaGrpID].append(pid)
        self.num_samples = data_source.num_spatialGroup
        self.spaGrpID_max = data_source.num_spatialGroup
        self.spaGrpID = 0

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        ret = []
        t_s = []
        for pid in np.unique(self.pid_dic[self.spaGrpID]):
            line_ids = np.nonzero(self.pid_dic[self.spaGrpID] == pid)[0].tolist()
            t = self.spaGrpID_dic[self.spaGrpID][line_ids[0]:line_ids[-1] + 1]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False).tolist()
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True).tolist()
            t_s += t
        ret.extend(t_s)
        self.spaGrpID += 1
        if self.spaGrpID == self.spaGrpID_max:
            self.spaGrpID = 0
        return iter(ret)
