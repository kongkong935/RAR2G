#!/bin/bash


delta_file="/root/autodl-tmp/sava/checkpoint_epoch2_step14067_bleu0.135087_cider0.244559.pth"
version="test是否工作"
savepath="/root/autodl-tmp/sava/$version"

python -u train.py \
    --test \
    --delta_file ${delta_file} \
    --batch_size 22 \
    --test_batch_size 50 \
    --freeze_vm True \
    --llm_use_lora False \
    --use_separate_queries True \
    --devices 1 \
    --val_check_interval 1.0 \
    --limit_test_batches 0.5 \
    --num_sanity_val_steps 1 \
    --savedmodel_path ${savepath} \
    2>&1 |tee -a ${savepath}/log.txt
