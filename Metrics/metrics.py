import torch
from torch import nn
from torch.nn import functional as F


def model_forward(model, inputs):
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super().__init__()
        boundaries = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer("bin_lowers", boundaries[:-1])
        self.register_buffer("bin_uppers", boundaries[1:])

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros((), device=logits.device)
        for lower, upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(lower) & confidences.le(upper)
            prop = in_bin.float().mean()
            if prop.item() > 0:
                ece = ece + (confidences[in_bin].mean() - accuracies[in_bin].float().mean()).abs() * prop
        return ece


class ClasswiseECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super().__init__()
        boundaries = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer("bin_lowers", boundaries[:-1])
        self.register_buffer("bin_uppers", boundaries[1:])

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        per_class = []
        for class_id in range(logits.size(1)):
            confidences = probs[:, class_id]
            targets = labels.eq(class_id).float()
            class_ece = torch.zeros((), device=logits.device)
            for lower, upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = confidences.gt(lower) & confidences.le(upper)
                prop = in_bin.float().mean()
                if prop.item() > 0:
                    class_ece = class_ece + (confidences[in_bin].mean() - targets[in_bin].mean()).abs() * prop
            per_class.append(class_ece)
        return torch.stack(per_class).mean()


@torch.no_grad()
def collect_logits(model, loader, device, max_batches=0):
    model.eval()
    logits_list = []
    labels_list = []
    for batch_idx, batch in enumerate(loader, start=1):
        inputs, labels = batch[0], batch[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits_list.append(model_forward(model, inputs))
        labels_list.append(labels)
        if max_batches and batch_idx >= max_batches:
            break
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def summarize_logits(logits, labels):
    predictions = logits.argmax(dim=1)
    return {
        "acc": predictions.eq(labels).float().mean().item(),
        "ece": ECELoss().to(logits.device)(logits, labels).item(),
        "cece": ClasswiseECELoss().to(logits.device)(logits, labels).item(),
        "nll": F.cross_entropy(logits, labels).item(),
    }

