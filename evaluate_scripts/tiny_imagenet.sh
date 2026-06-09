#!/bin/bash

# Tiny-ImageNet-C
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset tiny_imagenet \
    --data-root ./data/tiny-imagenet-200 \
    --model resnet50_ti \
    --checkpoint ./checkpoints/tiny_imagenet_resnet50_ti_fgr_seed1_best.pt \
    --batch-size 256 \
    --corruption-dataset tiny_imagenet_c \
    --corruption-root ./data/Tiny-ImageNet-C \
    --corruptions all \
    --severities 1 2 3 4 5 \
    --output-dir ./outputs
