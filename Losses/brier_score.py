from torch import nn
from torch.nn import functional as F


class BrierScore(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        target_probs = F.one_hot(targets, num_classes=logits.size(1)).to(dtype=probs.dtype)
        loss = (probs - target_probs).pow(2).sum(dim=1)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
