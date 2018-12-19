import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MetricNet(nn.Module):
    def __init__(self, feature_dim=256, num_class=0):
        super(MetricNet, self).__init__()
        self.num_class = num_class
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, self.num_class)
        init.normal_(self.out_layer.weight, std=0.001)
        init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        # feat = x[:, 0:-1]
        # motion_score = x[:, -1].view(-1, 1)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        # out = torch.cat((out, motion_score), dim=1)
        out = self.out_layer(out)
        return out


class AppearMotionNet(nn.Module):
    def __init__(self):
        super(AppearMotionNet, self).__init__()
        self.fc4 = nn.Linear(2, 2)

    def forward(self, x):
        out = self.fc4(x)
        return out
