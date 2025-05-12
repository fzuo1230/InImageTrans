import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from vary.model import *
from vary.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer,TextIteratorStreamer
from vary.model.plug.transforms import train_transform, test_transform
import json


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

def list_images(dir):
    all_images=[]
    for filename in os.listdir(dir):
        ext = os.path.splitext(filename)[1]
        if ext.lower() in ['.jpg','.png','.jpeg','webp','jfif']:
            all_images.append(dir+'/'+filename)
    return all_images


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = varyQwenForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device='cuda',  dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained("/mnt/home/user05/image/env/clip-vit-large-patch14/", torch_dtype=torch.float16)

    image_processor_high = test_transform

    use_im_start_end = True

    image_token_len = 256
    #print("that is a model")
    #model.eval()
    #print(model)
    with open("/mnt/home/user05/image/env/ChartQA-main/ChartQA Dataset/test/test_augmented.json","r",encoding="utf-8") as f:
        data = json.load(f)
    f.close()

    for it in data:
    
        
    #print("that is a model")

    # TODO download clip-vit in huggingface
    

    
    #qs = 'provide the OCR results of this image:'
    #qs = 'translate the OCR results into Chinese:'
        qs = it["query"]
    #qs = 'translate from english to Chinese.\nplease take nothing,but pictures,leave nothing but foot prints.'
    #print(qs)
        if use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN  + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


    

        conv_mode = "mpt"
        args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        inputs = tokenizer([prompt])

    #all_images = all_images[313:]
        image = load_image("/mnt/home/user05/image/env/ChartQA-main/ChartQA Dataset/test/png/" + it["imgname"])
        image_1 = image.copy()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        image_tensor_1 = image_processor_high(image_1)

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    #model.eval()

    #print(input_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
    #with torch.no_grad():
            output_ids = model.generate(
            input_ids,
            #images=None,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=True,
            num_beams = 1,
            temperature=0.0001,
            max_new_tokens=2048,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            #return_dict_in_generate=True,
            #output_hidden_states=True,
            #output_scores=True,
            #top_p=0.9,
            #top_k=30,
            #repetition_penalty=1.1
            #output_attentions=True
            )
        #print(output_ids)
            generate_text = ""
            for new_text in streamer:
                generate_text += new_text
            print(generate_text)
            with open("/mnt/home/user05/image/Vary-main/Vary-master/trans_result/chartqa","a",encoding="utf-8") as f:
                f.write(generate_text.replace("\n","")+'\n')
            f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/home/notebook/data/personal/S9055503/cotstage3_param/checkpoint-13000")
    parser.add_argument("--image-file", type=str, required=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()
    #with torch.no_grad():
    eval_model(args)