import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms

from .common import CIFAR10_MEAN, CIFAR10_STD, make_loader, resize_if_needed


NUM_CLASSES = 10
MEAN = CIFAR10_MEAN
STD = CIFAR10_STD


def get_train_valid_loader(
    root,
    batch_size,
    test_batch_size,
    valid_size=0.1,
    augment=True,
    num_workers=4,
    pin_memory=True,
    seed=1,
    download=False,
    input_size=None,
):
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    train_ops = []
    if augment:
        train_ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    train_ops.extend([transforms.ToTensor(), *resize_if_needed(input_size, 32), normalize])
    eval_ops = [transforms.ToTensor(), *resize_if_needed(input_size, 32), normalize]

    train_full = datasets.CIFAR10(root=root, train=True, download=download, transform=transforms.Compose(train_ops))
    val_full = datasets.CIFAR10(root=root, train=True, download=False, transform=transforms.Compose(eval_ops))
    indices = list(range(len(train_full)))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(np.floor(valid_size * len(indices)))
    val_set = Subset(val_full, indices[:split])
    train_set = Subset(train_full, indices[split:])
    train_loader = make_loader(train_set, batch_size, True, num_workers, pin_memory)
    val_loader = make_loader(val_set, test_batch_size, False, num_workers, pin_memory)
    return train_set, train_loader, val_loader


def get_test_loader(root, batch_size, num_workers=4, pin_memory=True, download=False, input_size=None):
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    ops = [transforms.ToTensor(), *resize_if_needed(input_size, 32), normalize]
    dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=transforms.Compose(ops))
    return make_loader(dataset, batch_size, False, num_workers, pin_memory)

