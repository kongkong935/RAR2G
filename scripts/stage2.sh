#!/bin/bash
export TOKENIZERS_PARALLELISM=false
#CUDA_VISIBLE_DEVICES=1
version="二阶段构建外部记忆库"
savepath="/root/autodl-tmp/Think in case/result/$version"
output_path="/root/autodl-tmp/Think in case/checkpoint"
annotation="/root/autodl-tmp/dataset/dataset_730.json"
stage1_checkpoint="/root/autodl-tmp/EVAP-autodl/checkpoint/stage1_checkpoint_epoch9_step6500.pth"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --stage 2 \
    --batch_size 100 \
    --freeze_vm True \
    --use_separate_queries False \
    --devices 2 \
    --learning_rate 1e-4 \
    --output_path ${output_path} \
    --annotation ${annotation} \
    --stage1_checkpoint ${stage1_checkpoint} \
    2>&1 |tee -a ${savepath}/log.txt
