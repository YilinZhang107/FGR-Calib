#!/bin/bash

# CE
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar100 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar100_resnet50_ce_seed1_best.pt \
    --batch-size 128 \
    --output-dir ./outputs \
    --output-name cifar100_ce_clean.json

# DFL
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar100 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar100_resnet50_dfl_seed1_best.pt \
    --batch-size 128 \
    --output-dir ./outputs \
    --output-name cifar100_dfl_clean.json

# FGR from scratch
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar100 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar100_resnet50_fgr_scratch_seed1_best.pt \
    --batch-size 128 \
    --corruption-dataset cifar100_c \
    --corruption-root ./data/CIFAR-100-C \
    --corruptions all \
    --severities 1 2 3 4 5 \
    --output-dir ./outputs \
    --output-name cifar100_fgr_scratch_c.json

# FGR fine-tune
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset cifar100 \
    --data-root ./data \
    --model resnet50 \
    --checkpoint ./checkpoints/cifar100_resnet50_fgr_finetune_seed1_best.pt \
    --batch-size 128 \
    --corruption-dataset cifar100_c \
    --corruption-root ./data/CIFAR-100-C \
    --corruptions all \
    --severities 1 2 3 4 5 \
    --output-dir ./outputs \
    --output-name cifar100_fgr_finetune_c.json
