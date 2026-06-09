import torch
from torch.nn import functional as F


def SoftBinnedECE(logits, labels, n_bins=15, temperature=0.1, p=2):
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    centers = torch.linspace(0.0, 1.0, n_bins, device=logits.device)
    distances = confidences.view(-1, 1) - centers.view(1, -1)
    weights = F.softmax(-(distances**2) / temperature, dim=1)

    bin_mass = weights.sum(dim=0) + 1e-8
    bin_conf = (confidences.unsqueeze(1) * weights).sum(dim=0) / bin_mass
    bin_acc = (accuracies.unsqueeze(1) * weights).sum(dim=0) / bin_mass
    errors = (bin_acc - bin_conf).abs().pow(p)
    return ((bin_mass / logits.size(0)) * errors).sum().pow(1.0 / p)

