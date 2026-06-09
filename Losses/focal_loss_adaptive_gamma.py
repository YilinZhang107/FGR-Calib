from torch import nn
from torch.nn import functional as F


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        adaptive_gamma = self.gamma + (1.0 - pt.detach()).clamp(0.0, 1.0)
        loss = -(1.0 - pt).pow(adaptive_gamma) * log_pt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
