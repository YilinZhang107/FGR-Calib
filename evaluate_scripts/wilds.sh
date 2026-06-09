#!/bin/bash

# iWildCam
CUDA_VISIBLE_DEVICES=0 python evaluate_wilds.py \
    --dataset iwildcam \
    --wilds-root ./data/wilds \
    --checkpoint ./checkpoints/iwildcam_resnet50_fgr_seed1_best.pt \
    --batch-size 64 \
    --splits val test \
    --output-dir ./outputs

# Camelyon17
CUDA_VISIBLE_DEVICES=0 python evaluate_wilds.py \
    --dataset camelyon17 \
    --wilds-root ./data/wilds \
    --checkpoint ./checkpoints/camelyon17_densenet121_fgr_seed1_best.pt \
    --batch-size 512 \
    --splits val test \
    --output-dir ./outputs

# FMoW
CUDA_VISIBLE_DEVICES=0 python evaluate_wilds.py \
    --dataset fmow \
    --wilds-root ./data/wilds \
    --checkpoint ./checkpoints/fmow_densenet121_fgr_seed1_best.pt \
    --batch-size 256 \
    --splits val test \
    --output-dir ./outputs
