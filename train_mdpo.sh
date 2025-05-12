#!/usr/bin/env python3
#!/bin/sh
export NCCL_P2P_LEVEL=NVL

llm="/mnt/home/user05/image/checkpoint-1000"
llm_model_path="/mnt/home/user05/image/checkpoint-1000"
deepspeed_config="/mnt/home/user05/image/Vary-main/Vary-master/zero_config/zero3_offload.json"
vision_tower="/mnt/home/user05/image/env/clip-vit-large-patch14"
report="train1.out"
output_path="/mnt/home/user05/image/env/nothing"


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
      --per_device_train_batch_size 1 \
      --num_train_epochs 0.2 \
      --learning_rate 1e-6 \
      --datasets 'mdpo_train' \
      --output_dir ${output_path}
