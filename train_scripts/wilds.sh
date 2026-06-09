#!/bin/bash

# FGR on iWildCam
CUDA_VISIBLE_DEVICES=0 python train_wilds.py \
    --dataset iwildcam \
    --wilds-root ./data/wilds \
    --pretrained \
    --loss dual_focal_loss \
    --gamma 7.0 \
    --lamda 0.1 \
    --epochs 10 \
    --batch-size 32 \
    --test-batch-size 64 \
    --lr 0.00003 \
    --weight-decay 0.0 \
    --fgr \
    --rho 0.05 \
    --start-epoch 0 \
    --filter-type jpeg_compression \
    --filter-severities 1 2 3 \
    --save-dir ./checkpoints \
    --output-dir ./outputs

# FGR on Camelyon17
CUDA_VISIBLE_DEVICES=0 python train_wilds.py \
    --dataset camelyon17 \
    --wilds-root ./data/wilds \
    --pretrained \
    --loss dual_focal_loss \
    --gamma 5.0 \
    --lamda 0.1 \
    --epochs 12 \
    --batch-size 512 \
    --test-batch-size 512 \
    --optimizer sgd \
    --lr 0.001 \
    --weight-decay 0.01 \
    --fgr \
    --rho 0.05 \
    --start-epoch 0 \
    --filter-type jpeg_compression \
    --filter-severities 1 2 3 \
    --save-dir ./checkpoints \
    --output-dir ./outputs

# FGR on FMoW
CUDA_VISIBLE_DEVICES=0 python train_wilds.py \
    --dataset fmow \
    --wilds-root ./data/wilds \
    --pretrained \
    --loss dual_focal_loss \
    --gamma 5.0 \
    --lamda 0.1 \
    --epochs 60 \
    --batch-size 64 \
    --test-batch-size 256 \
    --lr 0.0001 \
    --weight-decay 0.0 \
    --scheduler step \
    --step-gamma 0.96 \
    --fgr \
    --rho 0.05 \
    --start-epoch 0 \
    --filter-type jpeg_compression \
    --filter-severities 1 2 3 \
    --save-dir ./checkpoints \
    --output-dir ./outputs
