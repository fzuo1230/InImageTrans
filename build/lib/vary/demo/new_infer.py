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

from deep_model import DeepEraser
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import warnings
from paddleocr import PaddleOCR
from PIL import Image,ImageDraw,ImageFont
import numpy as np



import os
import requests
from PIL import Image
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer,TextIteratorStreamer
from vary.model.plug.transforms import train_transform, test_transform

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]

class x_BoxesConnector(object):
    def __init__(self, rects, imageW, max_dist=None, overlap_threshold=None):
        print('max_dist',max_dist)
        print('overlap_threshold',overlap_threshold )
        self.rects = np.array(rects)
        self.imageW = imageW
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageW)]  # 构建imageW个空列表
        for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[0]) < imageW:
                self.r_index[int(rect[0])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageW - 1].append(index)
        print(self.r_index)

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        print('y1', y1)
        Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)

        print('Yaxis_overlap', Yaxis_overlap)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        print('rect',rect)

        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            #print('left',left)
            for idx in self.r_index[left]:
                print('58796402',idx)
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:

                    return idx

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []       #相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any(): #优先级是not > and > or
                v = index
                print('v',v)
                sub_graphs.append([v])
                print('sub_graphs', sub_graphs)
                # 级联多个框(大于等于2个)
                print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():

                    v = np.where(self.graph[v, :])[0][0]          #np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    print('v11',v)
                    sub_graphs[-1].append(v)
                    print('sub_graphs11', sub_graphs)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):

            proposal = self.get_proposal(idx)
            print('idx11', idx)
            print('proposal',proposal)
            if proposal >= 0:

                self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1

        sub_graphs = self.sub_graphs_connected() #sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  #{0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])            #[[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:

            rect_set = self.rects[list(sub_graph)]     #[[228  78 238 128],[240  78 258 128]].....
            print('1234', rect_set)
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


class BoxesConnector(object):
    def __init__(self, rects, imageW, max_dist=5, overlap_threshold=0.2):
        self.rects = np.array(rects)
        self.imageW = imageW
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageW)]  # 构建imageW个空列表
        for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[1]) < imageW:
                self.r_index[int(rect[1])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageW - 1].append(index)
        print('self.r_index',self.r_index)
        print('len(self.r_index)', len(self.r_index))
    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)

        return Yaxis_overlap

    def calc_overlap_for_Xaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        width1 = self.rects[index1][2] - self.rects[index1][0]
        width2 = self.rects[index2][2] - self.rects[index2][0]
        x0 = max(self.rects[index1][0], self.rects[index2][0])
        x1 = min(self.rects[index1][2], self.rects[index2][2])

        Yaxis_overlap = max(0, x1 - x0) / max(width1, width2)
        print('Yaxis_overlap', Yaxis_overlap)
        return Yaxis_overlap


    def get_proposal(self, index):
        rect = self.rects[index]
        for left in range(rect[1] + 1, min(self.imageW - 1, rect[3] + self.max_dist)):
            for idx in self.r_index[left]:
                print('56871',idx)
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Xaxis(index, idx) > self.overlap_threshold:

                    return idx

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []       #相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any(): #优先级是not > and > or
                v = index
                print('v',v)
                sub_graphs.append([v])
                print('sub_graphs', sub_graphs)
                # 级联多个框(大于等于2个)
                print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():

                    v = np.where(self.graph[v, :])[0][0]          #np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    print('v11',v)
                    sub_graphs[-1].append(v)
                    print('sub_graphs11', sub_graphs)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):
            print('idx', idx)
            proposal = self.get_proposal(idx)

            print('proposal',proposal)
            if proposal > 0:

                self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1

        sub_graphs = self.sub_graphs_connected() #sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  #{0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])            #[[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:

            rect_set = self.rects[list(sub_graph)]     #[[228  78 238 128],[240  78 258 128]].....
            print('1234', rect_set)
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)

def get_rects(data):
    rects = []
    for it in data:
        rects.append([int(it[0][0]),int(it[0][1]),int(it[2][0]),int(it[2][1])])
    return rects

def get_y1(element):
    return element[1]

def merge_box(img_path,rects):
    img = Image.open(img_path).convert("RGB")
    x_connector = x_BoxesConnector(rects, img.size[0],max_dist=20,overlap_threshold=0.1)
    x_rects = x_connector.connect_boxes()
    connector = BoxesConnector(x_rects, img.size[1], max_dist=15, overlap_threshold=0.1)
    new_rects = connector.connect_boxes()
    sorted_indexes = np.argsort(new_rects[:,1])
    sorted_a = new_rects[sorted_indexes]
    new_rects = sorted_a
    ### get text_size
    text_size = []
    for it in new_rects:
        for item in rects:
            if it[1] == item[1]:
                text_size.append(item[3] - item[1])
                break
    return new_rects,text_size

def text_adding(img_path,new_rects,text_size,text):
    text = text.split("\n")
    ttf_path = "/mnt/home/user05/image/env/donut/synthdog/resources/font/zh/NotoSerifSC-Regular.otf"
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    if len(new_rects) == len(text):
        for rect,ts,chinese in zip(new_rects,text_size,text):
            font = ImageFont.truetype(ttf_path,ts)
            text_length = int(rect[2]-rect[0])
            refer = chinese
            x = rect[1]
            #print(img.size[1])
            while len(refer) > int(text_length/ts):
                if (x+2*ts)<=img.size[1]:
                    #print(x+2*ts)
                    draw.text((rect[0],x),refer[:int(text_length/ts)],font=font,fill="black")
                    refer = refer[int(text_length/ts):]
                    x = x + ts
                else:
                    break
            draw.text((rect[0],x),refer,font=font,fill="black")
    else:
        print("--------------text size---------------")
        print(text_size)
        ts = min(text_size)
        if ts<12:
            ts = 12
        font = ImageFont.truetype(ttf_path,ts)
        rect = new_rects[0]
        text_length = max([x[2]-x[0] for x in new_rects])
        init = min([x[0] for x in new_rects])
        #text_length = int(rect[2]-rect[0])
        x = rect[1]
        for chinese in text:
            refer = chinese
            while len(refer) > int(text_length/ts):
                if (x+2*ts)<=img.size[1]:
                    draw.text((init,x),refer[:int(text_length/ts)],font=font,fill="black")
                    refer = refer[int(text_length/ts):]
                    x = x+ts
                else:
                    break
            draw.text((rect[0],x),refer,font=font,fill="black")
            x = x+2*ts
    img.save("/mnt/home/user05/image/Vary-main/Vary-master/vary/demo/output_imgs/chinese.png")

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:1')
        # print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(rec_model_path, img_path, save_path):
    # print(torch.__version__)


    net = DeepEraser().cuda(1)


    reload_rec_model(net, rec_model_path)

    net.eval()

    img = np.array(Image.open(img_path+'input.jpg'))[:, :, :3]   
    mask = np.array(Image.open(img_path+'output.png'))[:, :]
    

    im = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask / 255.0).unsqueeze(0).float()
    
    with torch.no_grad():
    
        name = 'output'
        pred_img = net(im.unsqueeze(0).cuda(1), mask.unsqueeze(0).cuda(1))
        pred_img[-1] = torch.clamp(pred_img[-1], 0, 1)
        
        out = (pred_img[-1][0]*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        cv2.imwrite(save_path + name + '.png', out[:,:,::-1])
            

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def add_text(text,area,add_path):
    split_text = text.split("\n")
    print(split_text)
    ttf_path = "/mnt/home/user05/image/env/donut/synthdog/resources/font/zh/NotoSerifSC-Regular.otf"
    text_size = 20
    font = ImageFont.truetype(ttf_path, text_size)

    img = Image.open(add_path)
    draw = ImageDraw.Draw(img)
    
    for chinese,zone in zip(split_text,area):
        text_size = int(zone[3][1]-zone[0][1])
        text_length = int(zone[1][0]-zone[0][0])
        font = ImageFont.truetype(ttf_path, text_size)
        if len(chinese) > int(text_length/text_size):
           draw.text(zone[0],chinese[:int(text_length/text_size)],font=font,fill="black")
           split_text.insert(split_text.index(chinese)+1,chinese[int(text_length/text_size):])
           #split_text.append(chinese[int(text_length/text_size):])
        else:
            draw.text(zone[0],chinese,font=font,fill="black")
        

    img.save('/mnt/home/user05/image/Vary-main/Vary-master/vary/demo/output_imgs/chinese.png')


def draw_mask(img_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    img = Image.open(img_path).convert("RGB")
    img.save("/mnt/home/user05/image/Vary-main/Vary-master/vary/demo/input_imgs/input.jpg")
    canvas = Image.new('RGB',img.size,"black")
    draw = ImageDraw.Draw(canvas)
    result = ocr.ocr(img_path, cls=True)
    text_area = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            #print(line[0])
            area = line[0]
            new_area = []
            for it in area:
                new_area.append(tuple(it))
            text_area.append(new_area)
            draw.polygon(new_area,fill="white")


    canvas = canvas.convert('L')

    canvas.save('/mnt/home/user05/image/Vary-main/Vary-master/vary/demo/input_imgs/output.png')
    return text_area


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
    image_processor = CLIPImageProcessor.from_pretrained("/mnt/home/user05/image/env/clip-vit-large-patch14/", torch_dtype=torch.float16)

    image_processor_high = test_transform

    use_im_start_end = True

    image_token_len = 256

    
    #qs = 'provide the OCR results of this image:'
    #qs = 'get all the texts in the image:'
    qs = 'translate the OCR results into the Chinese.you can first ocr the image,then translate it.'
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


    inputs = tokenizer([prompt])


    image = load_image(args.image_file)
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
            #top_p=0.9,
            #top_k=30,
            temperature=0.0001,
            max_new_tokens=2048,
            streamer=streamer,
            repetition_penalty=1.1,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            #output_attentions=True
            )
        
        #print(output_ids.scores[0].shape)
        #probs=torch.softmax(output_ids.scores[0], dim=-1)
        #print(torch.max(probs))
        #print(output_ids)
        
        generate_text = ""
        for new_text in streamer:
            generate_text += new_text
        
        print(generate_text)
        #generate_text = "请不要拍摄任何东西，除了照片。什么都不要留下，除了脚印"
        rec_model_path = '/home/notebook/code/personal/S9055503/DeepEraser/deeperaser.pth'
        input_path = '/home/notebook/code/personal/S9055503/Vary-main/Vary-master/vary/demo/input_imgs/'
        img_path = args.image_file
        save_path =  '/home/notebook/code/personal/S9055503/Vary-main/Vary-master/vary/demo/output_imgs/'

        area = draw_mask(img_path)
        #print(area)
        
        rec(rec_model_path,input_path,save_path)
        bounding_box = get_rects(area)
        #print(bounding_box)
        new_rects,text_size = merge_box("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/vary/demo/output_imgs/output.png", bounding_box)
        text_size = [abs(x) for x in text_size]
        print("new_rects")
        print(new_rects)
        #print(text_size)
        #print(generate_text)
    
        text_adding("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/vary/demo/output_imgs/output.png",new_rects,text_size,generate_text)
        print(generate_text)
        ### bounding box merge

        #add_text(generate_text, area, "/home/notebook/code/personal/S9055503/Vary-main/Vary-master/vary/demo/output_imgs/output.png")
        


        '''
        print(type(output_ids.hidden_states))
        print(len(output_ids.hidden_states))
        print(type(output_ids.hidden_states[0]))
        print(len(output_ids.hidden_states[0]))
        print(type(output_ids.hidden_states[0][0]))
        print(output_ids.hidden_states[0][18].shape)
        #print(output_ids.hidden_states[1][0].tolist()[0][0])
        #token_res = list(output_ids.hidden_states[1])
        #print(list(output_ids.hidden_states[1]))
        '''

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
