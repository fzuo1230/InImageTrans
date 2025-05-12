#!/bin/sh

llm_model_path="/home/notebook/data/personal/S9055503/opt-125m"
deepspeed_config="zero_config/zero2.json"
vision_tower="/home/notebook/data/personal/S9055503/clip-vit-large-patch14"
datasets="data/Vary-600k/pdf_cn_30w.json"
report="train1.out"
pretrained_stage1_model="./model_param_2"
output_path="/home/notebook/data/personal/S9055503/stage_1"


deepspeed vary/train/train_opt.py \
      --deepspeed ${deepspeed_config} \
      --model_name_or_path ${llm_model_path} \
      --conversation_version "opt" \
      --use_cache  False \
      --vision_tower ${vision_tower} \
      --pretrained_stage1_model None \
      --freeze_vision_tower False \
      --freeze_lm_model False \
      --use_im_start_end True \
      --bf16 True \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 3000 \
      --save_total_limit 1 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --tf32 True \
      --model_max_length 2048 \
      --gradient_checkpointing True \
      --dataloader_num_workers 4 \
      --report_to wandb \
      --per_device_train_batch_size 16 \
      --num_train_epochs 3 \
      --learning_rate 1e-5 \
      --datasets "pdf_en" \
      --output_dir ${output_path}