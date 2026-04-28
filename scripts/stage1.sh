#!/bin/bash
export TOKENIZERS_PARALLELISM=false
#CUDA_VISIBLE_DEVICES=1
version="一阶段单正面qformer检查点"
savepath="/data/yz/sava/evap/$version"
annotation="/data/yz/Dataset/mimic_cxr/EVAP.json"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --stage 1 \
    --batch_size 100 \
    --freeze_vm True \
    --use_separate_queries False \
    --devices 2 \
    --learning_rate 1e-4 \
    --savedmodel_path ${savepath} \
    --annotation ${annotation} \
    2>&1 |tee -a ${savepath}/log.txt
