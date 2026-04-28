#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
version="阶段三+tok3+alpha0.7"
savepath="/root/autodl-tmp/Think in case/result/$version"
annotation="/root/autodl-tmp/dataset/EVAP_clean.json"
stage1_checkpoint="/root/autodl-tmp/EVAP-autodl/checkpoint/stage1_checkpoint_epoch9_step6500.pth"
ext_memory_path="/root/autodl-tmp/EVAP-autodl/checkpoint/ext_memory.pkl"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --stage 3 \
    --ext_memory_topn 3 \
    --ext_memory_alpha 0.7 \
    --batch_size 40 \
    --val_batch_size 25 \
    --freeze_vm True \
    --llm_use_lora False \
    --use_separate_queries True \
    --devices 1 \
    --learning_rate 1e-4 \
    --val_check_interval 0.5 \
    --limit_val_batches 1.0 \
    --num_sanity_val_steps 1 \
    --savedmodel_path "${savepath}" \
    --stage1_checkpoint "${stage1_checkpoint}" \
    --ext_memory_path "${ext_memory_path}" \
    2>&1 | tee -a "${savepath}/log.txt"
