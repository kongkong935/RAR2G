#!/bin/bash


delta_file="/root/autodl-tmp/checkpoint_epoch4_step14643_bleu0.137518_cider0.248795.pth"
version="三层对齐后加检索"
savepath="/root/autodl-tmp/sava/$version"

python -u train.py \
    --resume \
    --batch_size 5 \
    --val_batch_size 10 \
    --freeze_vm True \
    --llm_use_lora False \
    --use_separate_queries True \
    --devices 1 \
    --learning_rate 1e-4 \
    --val_check_interval 0.5 \
    --limit_val_batches 0.5 \
    --num_sanity_val_steps 1 \
    --delta_file ${delta_file} \
    --savedmodel_path ${savepath} \
    2>&1 |tee -a ${savepath}/log.txt
