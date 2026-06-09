import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from .common import IMAGENET_MEAN, IMAGENET_STD, make_loader, resize_if_needed


NUM_CLASSES = 200
MEAN = IMAGENET_MEAN
STD = IMAGENET_STD


class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        with open(self.root / "wnids.txt", "r", encoding="utf-8") as handle:
            classes = sorted(line.strip() for line in handle if line.strip())
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.samples = []
        with open(self.root / "val" / "val_annotations.txt", "r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().split("\t")
                if len(fields) >= 2:
                    self.samples.append((self.root / "val" / "images" / fields[0], class_to_idx[fields[1]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


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
    normalize = transforms.Normalize(MEAN, STD)
    train_ops = []
    if augment:
        train_ops.extend([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip()])
    train_ops.extend([transforms.ToTensor(), *resize_if_needed(input_size, 64), normalize])
    eval_ops = [transforms.ToTensor(), *resize_if_needed(input_size, 64), normalize]

    train_full = datasets.ImageFolder(str(Path(root) / "train"), transform=transforms.Compose(train_ops))
    val_full = datasets.ImageFolder(str(Path(root) / "train"), transform=transforms.Compose(eval_ops))
    indices = list(range(len(train_full)))
    random.Random(seed).shuffle(indices)
    split = int(valid_size * len(indices))
    val_set = Subset(val_full, indices[:split])
    train_set = Subset(train_full, indices[split:])
    train_loader = make_loader(train_set, batch_size, True, num_workers, pin_memory)
    val_loader = make_loader(val_set, test_batch_size, False, num_workers, pin_memory)
    return train_set, train_loader, val_loader


def get_test_loader(root, batch_size, num_workers=4, pin_memory=True, download=False, input_size=None):
    normalize = transforms.Normalize(MEAN, STD)
    ops = [transforms.ToTensor(), *resize_if_needed(input_size, 64), normalize]
    dataset = TinyImageNetVal(root=root, transform=transforms.Compose(ops))
    return make_loader(dataset, batch_size, False, num_workers, pin_memory)
