import os
import numpy as np
import matplotlib.pyplot as plt

'''
with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/reco_token.txt","r") as f:
    res = f.readlines()
f.close()
'''
with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/explain/trans_token.txt","r") as f:
    res = f.readlines()
f.close()

def compute_cosine(a_vec,b_vec):
    norms1 = np.linalg.norm(a_vec)
    norms2 = np.linalg.norm(b_vec)
    dot_products = np.sum(a_vec*b_vec)
    cos = dot_products / (norms1 * norms2)
    return cos

def cal_adj(lst):
    all_cos = []
    for it in lst:
        #print(len(it))
        adj_count = 0
        last = []
        while adj_count <= 32:
            cos = compute_cosine(it[adj_count], it[32])
            last.append(cos)
            adj_count += 1
        all_cos.append(last)
    return all_cos

def cal_last(lst):
    all_cos = []
    for it in lst:
        adj_count = 0
        last = []
        while adj_count < 32:
            cos = compute_cosine(it[adj_count], it[adj_count+1])
            last.append(cos)
            adj_count += 1
        all_cos.append(last)
    return all_cos

def cal_aver(lst):
    lst_num = len(lst[0])
    sum_lst = [0 for index in range(lst_num)]
    for it in lst:
        sum_lst = [x+y for x,y in zip(sum_lst,it)]
    sum_lst = [x/len(lst) for x in sum_lst]
    return sum_lst

def draw_pic(lst):
    x = list(range(1,len(lst)+1))
    plt.plot(x,lst,color="r",marker='o',markerfacecolor="blue")
    plt.title('analysis')
    plt.xlabel('layer')
    plt.ylabel('cosine_similarity')
    plt.savefig('./explain/trans_test.png')

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

all_token = str_to_list(res)
cos = cal_adj(all_token)
aver = cal_aver(cos)
#print(aver)
#draw_pic(aver)
cos2 = cal_last(all_token)
aver_cos2 = cal_aver(cos2)
draw_pic(aver)
#print(cos2)

'''
all_res = []
for it in res:
    it = it.replace("[", "").replace("]", "").replace("\n", "")
    item = it.split(",")
    item = [float(thing) for thing in item]
    item = np.array(item,dtype=float)
    all_res.append(item)

sim = []
for vec in all_res:
    sim1 = []
    for vec2 in all_res:
        cos = compute_cosine(vec, vec2)
        sim1.append(cos)
    sim.append(sim1)

adj = []
num = 0
while num < 32:
    cos = compute_cosine(all_res[num], all_res[num+1])
    adj.append(cos)
    num += 1
#print(adj)

with_lat = []
num2 = 0
while num2 <= 32:
    cos = compute_cosine(all_res[num2], all_res[32])
    with_lat.append(cos)
    num2 += 1
print(with_lat)
#print(sim)
#print(compute_cosine(all_res[1], all_res[1]))
'''


    