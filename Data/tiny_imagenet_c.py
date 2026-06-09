from pathlib import Path

from torchvision import datasets, transforms

from .common import IMAGENET_MEAN, IMAGENET_STD, make_loader, resize_if_needed


CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]


def get_test_loader(root, batch_size, corruption, severity, num_workers=4, pin_memory=True, input_size=None):
    corruption_root = Path(root) / corruption / str(severity)
    if not corruption_root.exists():
        raise FileNotFoundError(corruption_root)
    ops = [
        transforms.ToTensor(),
        *resize_if_needed(input_size, 64),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    dataset = datasets.ImageFolder(str(corruption_root), transform=transforms.Compose(ops))
    return make_loader(dataset, batch_size, False, num_workers, pin_memory)

