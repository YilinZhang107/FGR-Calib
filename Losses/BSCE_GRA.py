from torch.nn import functional as F

from .brier_score import BrierScore


def bsce_gra_loss(logits, targets, lamda=0.1):
    return F.cross_entropy(logits, targets) + lamda * BrierScore()(logits, targets)
