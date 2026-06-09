import torch
from torch import nn
from torch.nn import functional as F


def _mmce_kernel(conf_a, conf_b):
    return torch.exp(-torch.abs(conf_a.view(-1, 1) - conf_b.view(1, -1)) / 0.4)


class MMCE(nn.Module):
    def __init__(self, weighted=False, lamda=1.0):
        super().__init__()
        self.weighted = weighted
        self.lamda = lamda

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        correctness = predictions.eq(targets).float()
        errors = confidences - correctness
        weights = confidences if self.weighted else torch.ones_like(confidences)
        weighted_errors = errors * weights
        kernel = _mmce_kernel(confidences, confidences)
        value = (weighted_errors.view(1, -1) @ kernel @ weighted_errors.view(-1, 1)).squeeze()
        return self.lamda * torch.sqrt(torch.clamp(value, min=0.0)) / max(logits.size(0), 1)
