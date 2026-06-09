from torch.nn import functional as F

from .BSCE_GRA import bsce_gra_loss
from .brier_score import BrierScore
from .dual_focal_loss import DualFocalLoss
from .focal_loss import FocalLoss
from .focal_loss_adaptive_gamma import AdaptiveFocalLoss
from .mmce import MMCE
from .soft_ece import SoftBinnedECE


def normalize_loss_name(name):
    return (name or "cross_entropy").lower().replace("-", "_")


def compute_loss(logits, labels, name="cross_entropy", gamma=2.0, lamda=0.1):
    name = normalize_loss_name(name)
    if name in {"ce", "cross_entropy"}:
        return F.cross_entropy(logits, labels)
    if name in {"focal", "focal_loss"}:
        return FocalLoss(gamma=gamma)(logits, labels)
    if name in {"focal_adaptive", "focal_loss_adaptive", "adaptive_focal_loss"}:
        return AdaptiveFocalLoss(gamma=gamma)(logits, labels)
    if name in {"dual_focal", "dual_focal_loss", "dfl"}:
        return DualFocalLoss(gamma=gamma, reduction="mean")(logits, labels)
    if name == "mmce":
        return F.cross_entropy(logits, labels) + MMCE(weighted=False, lamda=lamda)(logits, labels)
    if name in {"mmce_weighted", "weighted_mmce"}:
        return F.cross_entropy(logits, labels) + MMCE(weighted=True, lamda=lamda)(logits, labels)
    if name in {"brier", "brier_score"}:
        return BrierScore()(logits, labels)
    if name in {"bsce", "bsce_gra"}:
        return bsce_gra_loss(logits, labels, lamda=lamda)
    raise ValueError(f"Unsupported loss: {name}")


def compute_calibration_loss(logits, labels, name="soft_ece", lamda=1.0):
    name = normalize_loss_name(name)
    if name in {"soft_ece", "soft_binned_ece"}:
        return lamda * SoftBinnedECE(logits, labels)
    if name == "mmce":
        return MMCE(weighted=False, lamda=lamda)(logits, labels)
    if name in {"mmce_weighted", "weighted_mmce"}:
        return MMCE(weighted=True, lamda=lamda)(logits, labels)
    if name in {"brier", "brier_score"}:
        return lamda * BrierScore()(logits, labels)
    raise ValueError(f"Unsupported calibration loss: {name}")
