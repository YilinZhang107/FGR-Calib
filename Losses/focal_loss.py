import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        loss = -(1.0 - pt).pow(self.gamma) * log_pt
        if self.alpha is not None:
            alpha = torch.as_tensor(self.alpha, dtype=loss.dtype, device=loss.device)
            if alpha.ndim == 0:
                loss = alpha * loss
            else:
                loss = alpha.gather(0, targets) * loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
