from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .cifar10_c import CORRUPTIONS
from .common import CIFAR100_MEAN, CIFAR100_STD, make_loader, resize_if_needed


class CIFAR100C(Dataset):
    def __init__(self, root, corruption, severity, input_size=None):
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in [1, 5]")
        root = Path(root)
        data_file = root / f"{corruption}.npy"
        labels_file = root / "labels.npy"
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not labels_file.exists():
            raise FileNotFoundError(labels_file)
        data = np.load(data_file, mmap_mode="r")
        labels = np.load(labels_file, mmap_mode="r")
        samples_per_severity = data.shape[0] // 5
        start = (severity - 1) * samples_per_severity
        end = severity * samples_per_severity
        self.data = data[start:end]
        self.labels = labels[start:end]
        self.transform = transforms.Compose(
            [transforms.ToTensor(), *resize_if_needed(input_size, 32), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(np.asarray(self.data[index]).astype(np.uint8)).convert("RGB")
        return self.transform(image), int(self.labels[index])


def get_test_loader(root, batch_size, corruption, severity, num_workers=4, pin_memory=True, input_size=None):
    dataset = CIFAR100C(root=root, corruption=corruption, severity=severity, input_size=input_size)
    return make_loader(dataset, batch_size, False, num_workers, pin_memory)

