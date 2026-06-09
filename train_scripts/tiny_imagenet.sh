#!/bin/bash

# FGR from scratch
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset tiny_imagenet \
    --data-root ./data/tiny-imagenet-200 \
    --model resnet50_ti \
    --loss dual_focal_loss \
    --gamma 3.0 \
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
    --output-dir ./outputs
