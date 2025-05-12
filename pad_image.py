import os
import json
import random
from PIL import Image,ImageDraw
#from paddleocr import PaddleOCR

def draw(img,area,count):
    image = Image.open(img)
    draw = ImageDraw.Draw(image)
    coord = []
    for box in area:
        a = tuple(box)
        coord.append(a)
    draw.polygon(coord,fill="white")
    del draw
    image.save('/home/notebook/data/personal/S9055503/pad_image/'+str(count)+".png")

def get_area(result):
    width = result[0]
    height = result[1]
    x = random.randint(int(width*0.1),int(width*0.5))
    y = random.randint(int(height*0.1),int(height*0.5))
    x_label,y_label = width*0.4,height*0.3
    area = [[x,y],[x+x_label,y],[x+x_label,y+y_label],[x,y+y_label]]
    return area

def get_all_path(text_path):
    with open(text_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    f.close()
    imgs = []
    for it in data:
        img = "/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/" + it["image"]
        imgs.append(img)
    return imgs

#ocr = PaddleOCR(se_angle_cls=True,lang="en")

imgs = get_all_path("/home/notebook/data/personal/S9055503/dpo_data/dpo_train_data.json")
count = 1
for img_path in imgs:
#img_path = "/home/notebook/data/personal/S9055503/test_image/image_7.jpg"
    #result = ocr.ocr(img_path,cls=True)
    img = Image.open(img_path)
    area = get_area(img.size)
    draw(img_path,area,count)
    print(count)
    count += 1 
