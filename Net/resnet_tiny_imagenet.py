from .resnet import Bottleneck, ResNet


def resnet50_tiny_imagenet(num_classes=200):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pool_size=2)
