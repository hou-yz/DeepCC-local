import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class MetricNet(nn.Module):
    def __init__(self, feature_dim=256, num_class=0):
        super(MetricNet, self).__init__()
        self.num_class = num_class

        layer_dim = feature_dim if feature_dim > 16 else 32

        self.fc1 = nn.Linear(feature_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.fc3 = nn.Linear(layer_dim, layer_dim)
        self.out_layer = nn.Linear(layer_dim, self.num_class)
        init.normal_(self.out_layer.weight, std=0.001)
        init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        # feat = x[:, 0:-1]
        # motion_score = x[:, -1].view(-1, 1)

        # dc_index = 4 * np.arange(256)#np.concatenate((4 * np.arange(256), 4 * np.arange(256) + 1))
        # x = x[:, dc_index]

        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        # out = torch.cat((out, motion_score), dim=1)
        out = self.out_layer(out)
        return out
