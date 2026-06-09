import torch
from torch import nn
from torch.nn import functional as F


class DualFocalLoss(nn.Module):
    def __init__(self, gamma=3.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        target_log_probs = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)

        non_target_probs = probs.clone()
        non_target_probs.scatter_(1, targets.view(-1, 1), -1.0)
        strongest_non_target = non_target_probs.max(dim=1).values.clamp_min(0.0)
        modulation = (1.0 - target_probs + strongest_non_target).pow(self.gamma)
        loss = -modulation * target_log_probs

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

