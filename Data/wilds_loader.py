from torchvision import transforms


WILDS_CONFIGS = {
    "iwildcam": {"image_size": (448, 448), "num_classes": 182, "num_workers": 4},
    "camelyon17": {"image_size": (96, 96), "num_classes": 2, "num_workers": 4},
    "fmow": {"image_size": (224, 224), "num_classes": 62, "num_workers": 4},
}


def get_wilds_transform(dataset_name):
    if dataset_name not in WILDS_CONFIGS:
        raise ValueError(f"Unsupported WILDS dataset: {dataset_name}")
    config = WILDS_CONFIGS[dataset_name]
    return transforms.Compose(
        [
            transforms.Resize(config["image_size"], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_wilds_loaders(dataset_name, root_dir, batch_size, test_batch_size, download=False, num_workers=None):
    from wilds import get_dataset
    from wilds.common.data_loaders import get_eval_loader, get_train_loader

    if dataset_name not in WILDS_CONFIGS:
        raise ValueError(f"Unsupported WILDS dataset: {dataset_name}")
    transform = get_wilds_transform(dataset_name)
    dataset = get_dataset(dataset=dataset_name, download=download, root_dir=root_dir)
    train_dataset = dataset.get_subset("train", transform=transform)
    val_dataset = dataset.get_subset("val", transform=transform)
    test_dataset = dataset.get_subset("test", transform=transform)
    workers = WILDS_CONFIGS[dataset_name]["num_workers"] if num_workers is None else int(num_workers)
    train_loader = get_train_loader("standard", train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    val_loader = get_eval_loader("standard", val_dataset, batch_size=test_batch_size, num_workers=workers, pin_memory=True)
    test_loader = get_eval_loader("standard", test_dataset, batch_size=test_batch_size, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader
