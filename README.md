# Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification

This repository contains the official implementation of **Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification**, accepted by **ICML 2026**.

Paper: [arXiv:2508.19830v2](https://arxiv.org/abs/2508.19830v2)

Frequency-Aware Gradient Rectification (FGR) trains with a small ratio of frequency-filtered samples and applies asymmetric projection-based gradient rectification to improve calibration under distribution shift.

## Supported Datasets

- CIFAR-10, CIFAR-100, CIFAR-10-C, CIFAR-100-C
- Tiny-ImageNet, Tiny-ImageNet-C
- WILDS: `iwildcam`, `camelyon17`, `fmow`

Datasets and checkpoints are not included in this repository.

## Installation

```bash
conda create -n fgr python=3.10 -y
conda activate fgr
pip install -r requirements.txt
```

SwanLab is optional and disabled by default.

## Dataset Layout

CIFAR-10/100 use the torchvision layout:

```text
/path/to/cifar/
  cifar-10-batches-py/
  cifar-100-python/
```

CIFAR-C roots should point directly to the `.npy` files:

```text
/path/to/CIFAR-10-C/
  gaussian_noise.npy
  labels.npy
```

Tiny-ImageNet uses the standard extracted layout:

```text
/path/to/tiny-imagenet-200/
  train/
  val/
  wnids.txt
```

Tiny-ImageNet-C should follow the corruption layout:

```text
/path/to/Tiny-ImageNet-C/
  gaussian_noise/1/<class>/*.JPEG
```

WILDS datasets should be prepared with the official `wilds` package.

## Two-Step Workflow

Training and corrupted evaluation are separated.

Train first:

```bash
bash train_scripts/cifar10.sh
```

Then evaluate the saved checkpoint:

```bash
bash evaluate_scripts/cifar10.sh
```

The scripts follow the original experiment-script style: each file contains multiple active commands. Comment out commands you do not want to run.

## Training Scripts

CIFAR-10:

```bash
bash train_scripts/cifar10.sh
```

CIFAR-100:

```bash
bash train_scripts/cifar100.sh
```

Tiny-ImageNet:

```bash
bash train_scripts/tiny_imagenet.sh
```

WILDS:

```bash
bash train_scripts/wilds.sh
```

The CIFAR-10/100 scripts include CE, Focal Loss, DFL, MMCE, FGR from scratch, and FGR fine-tuning from a CE checkpoint.

## Evaluation Scripts

CIFAR-10:

```bash
bash evaluate_scripts/cifar10.sh
```

CIFAR-100:

```bash
bash evaluate_scripts/cifar100.sh
```

Tiny-ImageNet-C:

```bash
bash evaluate_scripts/tiny_imagenet.sh
```

WILDS:

```bash
bash evaluate_scripts/wilds.sh
```

## Public Entry Points

You can also run the Python entry points directly.

```bash
python train.py --dataset cifar10 --data-root /path/to/cifar --model resnet50 --loss dual_focal_loss --gamma 5.0 --fgr
python evaluate.py --dataset cifar10 --data-root /path/to/cifar --checkpoint checkpoints/model.pt
python evaluate.py --dataset cifar10 --data-root /path/to/cifar --checkpoint checkpoints/model.pt --corruption-dataset cifar10_c --corruption-root /path/to/CIFAR-10-C
python train_wilds.py --dataset iwildcam --wilds-root /path/to/wilds --loss dual_focal_loss --gamma 7.0 --fgr
python evaluate_wilds.py --dataset iwildcam --wilds-root /path/to/wilds --checkpoint checkpoints/model.pt
```

## Losses

The training entry supports:

- `cross_entropy`
- `focal_loss`
- `focal_loss_adaptive`
- `dual_focal_loss`
- `mmce`
- `mmce_weighted`
- `brier_score`
- `bsce_gra`

FGR is enabled by passing `--fgr`. The main FGR hyperparameters are `--rho`, `--start-epoch`, `--filter-type`, `--filter-severities`, and `--calibration-loss`.

## Notes

- Edit paths and hyperparameters directly in the shell scripts before running.
- CIFAR FGR has two modes: from-scratch training for 350 epochs and fine-tuning from a CE checkpoint for 100 epochs.
- Checkpoints are saved under `./checkpoints` by default.
