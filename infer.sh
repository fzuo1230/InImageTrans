#!/bin/sh
model_name="/home/notebook/data/personal/S9055503/e2e_param/checkpoint-6000"
model_2="/home/notebook/data/personal/S9055503/only2_param/checkpoint-24000"
ocr_model="/mnt/home/user05/image/pretrained"
model_3="/home/notebook/data/personal/S9055503/only1_param/checkpoint-27000"
i2t="/home/notebook/data/personal/S9055503/reco_param/checkpoint-9000"

ocr_mt_model="/home/notebook/data/personal/S9055503/OCR+MT_param/checkpoint-12000"

image_file="/mnt/home/user05/image/env/real_Doc/documents/pybv0228_81.png"

python vary/demo/run_qwen_vary.py --model-name ${ocr_model} --image-file ${image_file}
