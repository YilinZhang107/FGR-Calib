from .densenet import densenet121
from .resnet import resnet50, resnet110
from .resnet_tiny_imagenet import resnet50_tiny_imagenet
from .wide_resnet import wide_resnet
from .wilds_initializer import create_wilds_model, freeze_backbone, get_wilds_optimizer_config


def create_cnn_model(name, num_classes, **kwargs):
    name = name.lower()
    if name == "resnet50":
        return resnet50(num_classes=num_classes)
    if name == "resnet50_ti":
        return resnet50_tiny_imagenet(num_classes=num_classes)
    if name == "resnet110":
        return resnet110(num_classes=num_classes)
    if name == "wide_resnet":
        return wide_resnet(
            num_classes=num_classes,
            depth=kwargs.get("depth", 26),
            width=kwargs.get("width", 10),
        )
    if name == "densenet121":
        return densenet121(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")
