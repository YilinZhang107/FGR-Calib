import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim

from Losses.losses import compute_calibration_loss, compute_loss
from Metrics.metrics import collect_logits, model_forward, summarize_logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def corrected_param_groups(model, correct_all=False):
    all_params = [p for p in model.parameters() if p.requires_grad]
    if correct_all:
        corrected = all_params
    elif hasattr(model, "fc"):
        corrected = [p for p in model.fc.parameters() if p.requires_grad]
    elif hasattr(model, "classifier"):
        corrected = [p for p in model.classifier.parameters() if p.requires_grad]
    elif hasattr(model, "head"):
        corrected = [p for p in model.head.parameters() if p.requires_grad]
    else:
        corrected = all_params
    corrected_ids = {id(p) for p in corrected}
    other = [p for p in all_params if id(p) not in corrected_ids]
    return corrected, other


def _flat_grads(grads, params, device):
    fixed = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    if not fixed:
        return torch.zeros(0, device=device)
    return torch.cat([g.reshape(-1) for g in fixed])


def apply_fgr_update(model, optimizer, loss_main, loss_calib, correct_all=False):
    corrected, other = corrected_param_groups(model, correct_all=correct_all)
    trainable = corrected + other
    if not trainable:
        raise RuntimeError("No trainable parameters found")
    if not corrected:
        optimizer.zero_grad(set_to_none=True)
        loss_main.backward()
        return 0.0

    device = trainable[0].device
    g_main_corr = torch.autograd.grad(loss_main, corrected, retain_graph=True, allow_unused=True)
    g_main_other = torch.autograd.grad(loss_main, other, retain_graph=True, allow_unused=True) if other else []
    g_calib_corr = torch.autograd.grad(loss_calib, corrected, retain_graph=False, allow_unused=True)

    main_flat = _flat_grads(g_main_corr, corrected, device)
    calib_flat = _flat_grads(g_calib_corr, corrected, device)
    dot = torch.dot(main_flat, calib_flat) if calib_flat.numel() else torch.tensor(0.0, device=device)
    if calib_flat.numel() and not torch.all(calib_flat == 0) and dot < 0:
        final_flat = main_flat - (dot / (torch.dot(calib_flat, calib_flat) + 1e-12)) * calib_flat
    else:
        final_flat = main_flat

    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for param in corrected:
        numel = param.numel()
        param.grad = final_flat[offset : offset + numel].view_as(param).detach().clone()
        offset += numel
    for grad, param in zip(g_main_other, other):
        param.grad = (torch.zeros_like(param) if grad is None else grad).detach().clone()
    return float(dot.detach().cpu().item())


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_name="dual_focal_loss",
    gamma=2.0,
    lamda=0.1,
    use_fgr=False,
    calibration_loss="soft_ece",
    calibration_weight=1.0,
    correct_all=False,
    max_steps=0,
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    dots = []

    for step, batch in enumerate(loader, start=1):
        inputs, labels = batch[0], batch[1]
        if len(batch) >= 3 and torch.is_tensor(batch[2]) and batch[2].dtype == torch.bool:
            is_corrupted = batch[2]
        else:
            is_corrupted = torch.zeros(len(labels), dtype=torch.bool)
        clean_mask = (~is_corrupted).to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model_forward(model, inputs)
        loss_main = compute_loss(logits, labels, name=loss_name, gamma=gamma, lamda=lamda)
        if use_fgr and torch.sum(clean_mask) > 0:
            loss_calib = compute_calibration_loss(
                logits[clean_mask],
                labels[clean_mask],
                name=calibration_loss,
                lamda=calibration_weight,
            )
            dot = apply_fgr_update(model, optimizer, loss_main, loss_calib, correct_all=correct_all)
            dots.append(dot)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss_main.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += float(loss_main.detach().cpu().item())
        total_steps += 1
        if max_steps and step >= max_steps:
            break

    return {"loss": total_loss / max(total_steps, 1), "avg_dot": float(np.mean(dots)) if dots else 0.0}


def evaluate(model, loader, device, max_batches=0):
    logits, labels = collect_logits(model, loader, device, max_batches=max_batches)
    return summarize_logits(logits, labels)


def build_optimizer(model, name, lr, weight_decay, momentum=0.9, nesterov=False):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, config):
    name = (config.get("scheduler") or "multistep").lower()
    if name in {"none", "disabled"}:
        return None
    if name == "multistep":
        milestones = config.get("milestones") or []
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=config.get("lr_decay", 0.1))
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 1),
            gamma=config.get("step_gamma", config.get("lr_decay", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def load_checkpoint(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, config=None, metrics=None):
    ensure_dir(os.path.dirname(path))
    payload = {"state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if config is not None:
        payload["config"] = config
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, path)


def save_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class ExperimentLogger:
    def __init__(self, mode="disabled", project="fgr", run_name=None, config=None):
        self.enabled = mode != "disabled"
        self.swanlab = None
        if self.enabled:
            try:
                import swanlab

                self.swanlab = swanlab
                swanlab.init(project=project, experiment_name=run_name, config=config, mode=mode)
            except Exception as exc:
                print(f"Warning: SwanLab logging disabled: {exc}")
                self.enabled = False

    def log(self, values, step=None):
        if self.enabled and self.swanlab is not None:
            self.swanlab.log(values, step=step)

    def finish(self):
        if self.enabled and self.swanlab is not None:
            self.swanlab.finish()
