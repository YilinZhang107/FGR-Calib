from torch import nn
from torchvision import models as tv_models


def create_wilds_model(dataset_name, num_classes, pretrained=True):
    dataset_name = dataset_name.lower()
    if dataset_name == "iwildcam":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if dataset_name in {"camelyon17", "fmow"}:
        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported WILDS dataset: {dataset_name}")


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            for parameter in getattr(model, attr).parameters():
                parameter.requires_grad = True
            return
    raise ValueError("Could not find a classifier head to unfreeze")


def get_wilds_optimizer_config(dataset_name):
    configs = {
        "iwildcam": {
            "optimizer": "adam",
            "lr": 3e-5,
            "weight_decay": 0.0,
            "epochs": 10,
            "batch_size": 32,
            "test_batch_size": 64,
        },
        "camelyon17": {
            "optimizer": "sgd",
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 1e-2,
            "epochs": 12,
            "batch_size": 512,
            "test_batch_size": 512,
        },
        "fmow": {
            "optimizer": "adam",
            "lr": 1e-4,
            "weight_decay": 0.0,
            "epochs": 60,
            "batch_size": 64,
            "test_batch_size": 256,
            "step_gamma": 0.96,
        },
    }
    if dataset_name not in configs:
        raise ValueError(f"Unsupported WILDS dataset: {dataset_name}")
    return configs[dataset_name]
