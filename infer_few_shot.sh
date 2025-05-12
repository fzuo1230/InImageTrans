#!/bin/sh
model_name="/home/notebook/data/personal/S9055503/e2e_param/checkpoint-6000"
pretrained_model="/home/notebook/data/personal/S9055503/pretrained_model"
image_file="/home/notebook/data/personal/S9055503/test_image/test10.jpg"
python vary/demo/few_shot.py --model-name ${pretrained_model} --image-file ${image_file}