#!/usr/bin/env python3
#!/bin/sh
export NCCL_P2P_LEVEL=NVL
export PATH=$PATH:~/.local/bin

gpu_ids=0,1
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)

llm="/home/notebook/data/personal/S9055503/cotstage3_param/checkpoint-13000"
llm_model_path="/home/notebook/data/personal/S9055503/transstage3_param/checkpoint-13000"
deepspeed_config="/home/notebook/code/personal/S9055503/Vary-main/Vary-master/zero_config/zero2.json"
vision_tower="/home/notebook/data/personal/S9055503/clip-vit-large-patch14"
pretrained_stage1_model="/home/notebook/data/personal/S9055503/Varytiny-600k.pth"
datasets="data/Vary-600k"
report="train1.out"
output_path="/home/notebook/data/personal/S9055503/dpo_param"


deepspeed vary/train/train_dpo.py \
      --deepspeed ${deepspeed_config} \
      --model_name_or_path ${llm} \
      --use_cache  False \
      --vision_tower ${vision_tower} \
      --freeze_vision_tower True \
      --freeze_lm_model False \
      --pretrained_stage1_model None \
      --vision_select_layer -2 \
      --use_im_start_end True \
      --bf16 True \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 500 \
      --save_total_limit 1 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --tf32 True \
      --model_max_length 4096 \
      --gradient_checkpointing True \
      --dataloader_num_workers 1 \
      --report_to none \
      --per_device_train_batch_size 2 \
      --num_train_epochs 0.2 \
      --learning_rate 1e-6 \
      --datasets 'dpo_train' \
      --output_dir ${output_path}
