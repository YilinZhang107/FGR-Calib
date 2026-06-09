import math

import torch
from torch import nn
from torch.nn import functional as F


class DenseBottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([out, x], dim=1)


class DenseTransition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)

    def forward(self, x):
        return F.avg_pool2d(self.conv(F.relu(self.bn(x), inplace=True)), 2)


class DenseNet(nn.Module):
    def __init__(self, blocks, growth_rate=32, reduction=0.5, num_classes=10):
        super().__init__()
        planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, planes, 3, padding=1, bias=False)
        self.dense1, planes = self._dense_layer(planes, blocks[0], growth_rate)
        self.trans1, planes = self._transition(planes, reduction)
        self.dense2, planes = self._dense_layer(planes, blocks[1], growth_rate)
        self.trans2, planes = self._transition(planes, reduction)
        self.dense3, planes = self._dense_layer(planes, blocks[2], growth_rate)
        self.trans3, planes = self._transition(planes, reduction)
        self.dense4, planes = self._dense_layer(planes, blocks[3], growth_rate)
        self.bn = nn.BatchNorm2d(planes)
        self.fc = nn.Linear(planes, num_classes)
        self.architecture = "CNN"

    def _dense_layer(self, in_planes, nblocks, growth_rate):
        layers = []
        planes = in_planes
        for _ in range(nblocks):
            layers.append(DenseBottleneck(planes, growth_rate))
            planes += growth_rate
        return nn.Sequential(*layers), planes

    def _transition(self, in_planes, reduction):
        out_planes = int(math.floor(in_planes * reduction))
        return DenseTransition(in_planes, out_planes), out_planes

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out), inplace=True), (1, 1))
        features = out.view(out.size(0), -1)
        return self.fc(features), features


def densenet121(num_classes):
    return DenseNet([6, 12, 24, 16], num_classes=num_classes)
