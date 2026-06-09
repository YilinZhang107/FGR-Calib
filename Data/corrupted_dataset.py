import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .corruptions import apply_corruption, random_corruption


class CorruptedDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        corruption_prob=0.05,
        corruption_types=None,
        severities=None,
        mean=None,
        std=None,
        return_index=False,
    ):
        self.base_dataset = base_dataset
        self.corruption_prob = corruption_prob
        self.corruption_types = corruption_types or ["jpeg_compression"]
        self.severities = severities or [1, 2, 3]
        self.mean = torch.tensor(mean or [0.0, 0.0, 0.0]).view(3, 1, 1)
        self.std = torch.tensor(std or [1.0, 1.0, 1.0]).view(3, 1, 1)
        self.return_index = return_index
        self.reset_corruption_selection()

    def reset_corruption_selection(self):
        total = len(self.base_dataset)
        num_corrupted = int(total * self.corruption_prob)
        if num_corrupted <= 0:
            self.corrupted_indices = set()
        else:
            self.corrupted_indices = set(random.sample(range(total), num_corrupted))

    def __len__(self):
        return len(self.base_dataset)

    def _tensor_to_pil(self, image):
        image = image.detach().cpu()
        if image.ndim != 3:
            raise ValueError("expected CHW image tensor")
        restored = image * self.std + self.mean
        restored = restored.clamp(0.0, 1.0)
        array = (restored.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(array)

    def _pil_to_tensor(self, image, like_tensor):
        tensor = transforms.ToTensor()(image)
        mean = self.mean.to(tensor.device, dtype=tensor.dtype)
        std = self.std.to(tensor.device, dtype=tensor.dtype)
        tensor = (tensor - mean) / std
        return tensor.to(dtype=like_tensor.dtype)

    def _split_sample(self, sample):
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("base dataset must return at least (image, label)")
        return sample[0], sample[1]

    def __getitem__(self, index):
        image, label = self._split_sample(self.base_dataset[index])
        is_corrupted = index in self.corrupted_indices

        if is_corrupted:
            corruption_type, severity = random_corruption(self.corruption_types, self.severities)
            if isinstance(image, torch.Tensor):
                pil_image = self._tensor_to_pil(image)
                image = self._pil_to_tensor(apply_corruption(pil_image, corruption_type, severity), image)
            else:
                image = apply_corruption(image, corruption_type, severity)

        if self.return_index:
            return image, label, is_corrupted, index
        return image, label, is_corrupted

