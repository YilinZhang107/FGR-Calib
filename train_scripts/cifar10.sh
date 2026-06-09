#!/bin/bash

# CE
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --loss cross_entropy \
    --gamma 1.0 \
    --lamda 0.1 \
    --epochs 350 \
    --batch-size 128 \
    --test-batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.0005 \
    --milestones 150 250 \
    --save-dir ./checkpoints \
    --output-dir ./outputs \
    --run-name cifar10_resnet50_ce_seed1

# Focal Loss
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --loss focal_loss \
    --gamma 3.0 \
    --lamda 0.1 \
    --epochs 350 \
    --batch-size 128 \
    --test-batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.0005 \
    --milestones 150 250 \
    --save-dir ./checkpoints \
    --output-dir ./outputs \
    --run-name cifar10_resnet50_focal_seed1

# DFL
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --loss dual_focal_loss \
    --gamma 5.0 \
    --lamda 0.1 \
    --epochs 350 \
    --batch-size 128 \
    --test-batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.0005 \
    --milestones 150 250 \
    --save-dir ./checkpoints \
    --output-dir ./outputs \
    --run-name cifar10_resnet50_dfl_seed1

# MMCE
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --loss mmce_weighted \
    --gamma 1.0 \
    --lamda 2.0 \
    --epochs 350 \
    --batch-size 128 \
    --test-batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.0005 \
    --milestones 150 250 \
    --save-dir ./checkpoints \
    --output-dir ./outputs \
    --run-name cifar10_resnet50_mmce_seed1

# FGR from scratch
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --loss dual_focal_loss \
    --gamma 5.0 \
    --lamda 0.1 \
    --epochs 350 \
    --start-epoch 200 \
    --batch-size 128 \
    --test-batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.0005 \
    --milestones 150 250 \
    --fgr \
    --rho 0.05 \
    --filter-type jpeg_compression \
    --filter-severities 1 2 3 \
    --save-dir ./checkpoints \
    --output-dir ./outputs \
    --run-name cifar10_resnet50_fgr_scratch_seed1

# FGR fine-tune
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar10 \
    --data-root ./data \
    --model resnet50 \
    --loss dual_focal_loss \
    --gamma 5.0 \
    --lamda 0.1 \
    --epochs 100 \
    --start-epoch 0 \
    --batch-size 128 \
    --test-batch-size 256 \
    --lr 0.01 \
    --weight-decay 0.0005 \
    --milestones 40 70 \
    --fgr \
    --rho 0.05 \
    --filter-type jpeg_compression \
    --filter-severities 1 2 3 \
    --load \
    --checkpoint ./checkpoints/cifar10_resnet50_ce_seed1_best.pt \
    --freeze \
    --save-dir ./checkpoints \
    --output-dir ./outputs \
    --run-name cifar10_resnet50_fgr_finetune_seed1
