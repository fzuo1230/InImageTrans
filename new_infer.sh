#!/bin/sh
model_name="/home/notebook/data/personal/S9055503/e2e_param/checkpoint-6000"
model_2="/home/notebook/data/personal/S9055503/only2_param/checkpoint-24000"
ocr_model="/mnt/home/user05/image/pretrained"
model_3="/home/notebook/data/personal/S9055503/only1_param/checkpoint-27000"
i2t="/home/notebook/data/personal/S9055503/reco_param/checkpoint-9000"

ocr_mt_model="/home/notebook/data/personal/S9055503/valid_param/checkpoint-3000"

multi_model="/home/notebook/data/personal/S9055503/multi_task_param/checkpoint-24000"

stage3_model="/home/notebook/data/personal/S9055503/transstage3_param_cont/checkpoint-8000"

stage3_model2="/home/notebook/data/personal/S9055503/cotstage3_param/checkpoint-13000"

stage3_model2_add="/home/notebook/data/personal/S9055503/addtransstage3_param/checkpoint-13000"

dpo_model="/mnt/home/user05/image/checkpoint-1000"

image_file="/mnt/home/user05/image/Vary-main/assets/vary.png"

python vary/demo/eval.py --model-name ${ocr_model} --image-file ${image_file}
