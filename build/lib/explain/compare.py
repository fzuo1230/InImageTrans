import os
import numpy as np
import seaborn

with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/all_token.txt","r") as f:
    res = f.readlines()
f.close()

with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/trans_token.txt","r") as f:
    trans = f.readlines()
f.close()

def compute_cosine(a_vec,b_vec):
    norms1 = np.linalg.norm(a_vec)
    norms2 = np.linalg.norm(b_vec)
    dot_products = np.sum(a_vec*b_vec)
    cos = dot_products / (norms1 * norms2)
    return cos

def str_to_list(lst):
    #print(len(lst))
    reg_num = 1
    all_token = []
    token = []
    for it in lst:
        item = it.replace("[", "").replace("]", "").replace("\n", "")
        item = item.split(",")
        item = [float(thing) for thing in item]
        item = np.array(item,dtype=float)
        if reg_num <=33 :
            #print(reg_num,lst.index(it))
            token.append(item)
            reg_num += 1
        else:
            all_token.append(token)
            token = []
            reg_num = 2
            token.append(item)
    #print(reg_num)
    all_token.append(token)
    return all_token

def get_aver(lst):
    lst_num = len(lst[0][0])
    #print(lst_num)
    aver_token = np.zeros((33,4096))
    #print(aver_token)
    for it in lst:
        num = 0
        while num < 33:
            aver_token[num] = np.add(aver_token[num],it[num])
            num += 1
    #print(len(lst))
    #print(aver_token/len(lst))
    return aver_token

def draw_pic(lst):
    x = list(range(0,33))
    y = list(range(0,33))
    ax = seaborn.heatmap(lst,xticklabels=x,yticklabels=y,annot=False)
    ax.set_title('heatmap')
    ax.set_xlabel('reco')
    ax.set_ylabel('trans')
    figure = ax.get_figure()
    figure.savefig('./explain/pro_heatmap.jpg')

def norm_cos(lst):
    a,b = 0,0
    while a < 33:
        while b < 33:
            if lst[a][b] < 0.9:
                lst[a][b] = 0
            b += 1
        b = 0
        a += 1
    return lst

all_token = str_to_list(res)
#print(get_aver(all_token))

trans_token = str_to_list(trans)

all_token = get_aver(all_token)
#print(all_token)
trans_token = get_aver(trans_token)
#print(trans_token)

rows = 33
cols = 33
all_cos = [[0 for _ in range(cols)] for _ in range(rows)]

x,y = 0,0
while x < rows:
    while y < cols:
        all_cos[x][y] = compute_cosine(all_token[x], trans_token[y])
        y += 1
    y = 0
    x += 1
all_cos = norm_cos(all_cos)
draw_pic(all_cos)
#print(all_cos)
    