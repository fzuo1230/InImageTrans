import evaluate
import os

with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/trans_result/output_document_mdpo_repe","r",encoding="utf-8") as f:
    out = f.readlines()
f.close()

with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/trans_result/ref.txt","r",encoding="utf-8") as f:
    ref = f.readlines()
f.close()

meteor = evaluate.load('/home/notebook/code/personal/S9055503/Vary-main/Vary-master/trans_result/meteor.py')


#pred = ["我喜欢你"]
#refer = ["我喜欢你"]
results = meteor.compute(predictions=out,references=ref)
print(round(results['meteor'],4))