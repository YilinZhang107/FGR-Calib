import os

from torch.utils.data import DataLoader
from torchvision import transforms


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
CIFAR100_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR100_STD = [0.2023, 0.1994, 0.2010]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resize_if_needed(input_size, native_size):
    if input_size and int(input_size) != native_size:
        return [transforms.Resize((int(input_size), int(input_size)))]
    return []


def make_loader(dataset, batch_size, shuffle, num_workers=4, pin_memory=True):
    workers = max(int(num_workers), 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def resolve_root(value, env_name, default):
    if value:
        return value
    return os.environ.get(env_name, default)

