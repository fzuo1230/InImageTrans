import os
import json

with open("/home/notebook/data/personal/S9055503/multi_prompt/multi_only_trans.json","r",encoding="utf-8") as f:
    data= json.load(f)
f.close()

print(data[0])