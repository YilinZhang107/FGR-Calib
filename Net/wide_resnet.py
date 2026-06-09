import torch
from torch import nn
from torch.nn import functional as F


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class WideResNet(nn.Module):
    def __init__(self, depth=26, width=10, num_classes=10):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy (depth - 2) % 6 == 0")
        n = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.layer1 = self._make_layer(16 * width, n, 1)
        self.layer2 = self._make_layer(32 * width, n, 2)
        self.layer3 = self._make_layer(64 * width, n, 2)
        self.bn = nn.BatchNorm2d(64 * width)
        self.fc = nn.Linear(64 * width, num_classes)
        self.architecture = "CNN"

    def _make_layer(self, planes, blocks, stride):
        layers = [WideBasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(WideBasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        features = out.view(out.size(0), -1)
        return self.fc(features), features


def wide_resnet(num_classes, depth=26, width=10):
    return WideResNet(depth=depth, width=width, num_classes=num_classes)
