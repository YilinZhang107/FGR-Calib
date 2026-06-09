from .corrupted_dataset import CorruptedDataset
from . import cifar10, cifar100, cifar10_c, cifar100_c, tiny_imagenet, tiny_imagenet_c


DATASET_MODULES = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "tiny_imagenet": tiny_imagenet,
}

CORRUPTION_MODULES = {
    "cifar10_c": cifar10_c,
    "cifar100_c": cifar100_c,
    "tiny_imagenet_c": tiny_imagenet_c,
}


def get_dataset_module(name):
    if name not in DATASET_MODULES:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_MODULES[name]


def get_corruption_module(name):
    if name not in CORRUPTION_MODULES:
        raise ValueError(f"Unsupported corruption dataset: {name}")
    return CORRUPTION_MODULES[name]
