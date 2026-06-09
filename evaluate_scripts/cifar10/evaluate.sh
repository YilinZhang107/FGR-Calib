#!/bin/bash

# CE
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar10_resnet50_ce_seed1_best.pt \
    --batch-size 256 \
    --output-dir ./outputs \
    --output-name cifar10_ce_clean.json

# DFL
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar10_resnet50_dfl_seed1_best.pt \
    --batch-size 256 \
    --output-dir ./outputs \
    --output-name cifar10_dfl_clean.json

# FGR from scratch
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar10_resnet50_fgr_scratch_seed1_best.pt \
    --batch-size 256 \
    --corruption-dataset cifar10_c \
    --corruption-root ./data/CIFAR-10-C \
    --corruptions all \
    --severities 1 2 3 4 5 \
    --output-dir ./outputs \
    --output-name cifar10_fgr_scratch_c.json

# FGR fine-tune
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar10_resnet50_fgr_finetune_seed1_best.pt \
    --batch-size 256 \
    --corruption-dataset cifar10_c \
    --corruption-root ./data/CIFAR-10-C \
    --corruptions all \
    --severities 1 2 3 4 5 \
    --output-dir ./outputs \
    --output-name cifar10_fgr_finetune_c.json
