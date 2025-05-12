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
from transformers import TextStreamer
from vary.model.plug.transforms import train_transform, test_transform

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


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
    #print("that is a model")
    #model.eval()
    #print(model)
    '''
    total_num = sum(p.numel() for n, p in model.named_parameters())
    print(total_num)
    '''
    model.to(device='cuda',  dtype=torch.bfloat16)
    print("that is a model")

    # TODO download clip-vit in huggingface
    image_processor = CLIPImageProcessor.from_pretrained("/home/notebook/data/personal/S9055503/clip-vit-large-patch14/", torch_dtype=torch.float16)

    image_processor_high = test_transform

    use_im_start_end = True

    image_token_len = 256

    
    #qs = 'provide the OCR results of this image:'
    #qs = 'get all the texts in the image:'
    qs = 'translate the OCR results into the Chinese:'
    #qs = 'translate from english to Chinese.\nplease take nothing,but pictures,leave nothing but foot prints.'
    print(qs)
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
    ###shot
    answer = '请除了照片什么都别拿走，除了脚印什么都别留下'
    conv_shot = conv_templates[args.conv_mode].copy()
    conv_shot.append_message(conv.roles[0], qs)
    conv_shot.append_message(conv.roles[1], answer)
    prompt_shot = conv_shot.get_prompt()
    ###shot
    prompt = prompt.replace("<|im_start|>system\nYou should follow the instructions carefully and explain your answers in detail.<|im_end|>","")
    #print(prompt)
    #print(prompt_shot)
    #print(type(prompt_shot))
    prompt = prompt_shot+prompt
    print(prompt)
    inputs = tokenizer([prompt])
    #print(len(inputs.input_ids[0]))
    #print(len(inputs.input_ids[1]))
    ###few-shot
    image_shot = load_image("/home/notebook/data/personal/S9055503/test_image/test7.jpg")
    image_shot1 = image_shot.copy()
    image_shot_tensor = image_processor.preprocess(image_shot1,return_tensors='pt')['pixel_values'][0]
    image_shot_tensor_1 = image_processor_high(image_shot1)
    ###few-shot

    image = load_image(args.image_file)
    image_1 = image.copy()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #print(image_tensor.shape)

    image_tensor_1 = image_processor_high(image_1)
    #print(image_tensor_1.shape)
    
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    #print(input_ids)
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    #model.eval()

    print(input_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
    #with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            #images=None,
            #images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda()),(image_shot_tensor.unsqueeze(0).half().cuda(), image_shot_tensor_1.unsqueeze(0).half().cuda())],
            images=[(image_shot_tensor.unsqueeze(0).half().cuda(), image_shot_tensor_1.unsqueeze(0).half().cuda()),(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=True,
            num_beams = 1,
            temperature=0.0001,
            max_new_tokens=2048,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            #output_attentions=True
            )
        print(type(output_ids.hidden_states))
        print(len(output_ids.hidden_states))
        print(type(output_ids.hidden_states[0]))
        print(len(output_ids.hidden_states[0]))
        print(type(output_ids.hidden_states[0][0]))
        print(output_ids.hidden_states[0][18].shape)
        #print(output_ids.hidden_states[1][0].tolist()[0][0])
        #token_res = list(output_ids.hidden_states[1])
        #print(list(output_ids.hidden_states[1]))
        
        num = 0
        #all_token_res = []
        '''
        while num < len(output_ids.hidden_states)-1:
            token_res = output_ids.hidden_states[num+1]
            for token in token_res:
               tokens = token.tolist()[0][0]
               with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/trans_token.txt","a") as f:
                  f.write(str(tokens)+"\n")
               f.close()
            #all_token_res.append(all_token)
            num += 1
        
        '''
            #print(token)
        
        #print(token_res)
        '''
        for state in output_ids.hidden_states[0]:
            res = state.tolist()[0]
            #print(len(res))
            #print(res)
            with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/reco_token.txt","a") as f:
                f.write(str(res)+"\n")
            f.close()
        '''
        
            #with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/reco_token.txt","a") as f:

        #with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/prompt.txt","a") as f:

        
        # print(output_ids)

        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        # # conv.messages[-1][-1] = outputs
        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        # outputs = outputs.strip()

        # print(outputs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()
    #with torch.no_grad():
    eval_model(args)
