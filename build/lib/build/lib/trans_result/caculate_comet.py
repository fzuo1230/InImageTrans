from comet import download_model, load_from_checkpoint

# Choose your model from Hugging Face Hub

#model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
# model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint("/home/notebook/data/personal/S9055503/XCOMET-XL/checkpoints/model.ckpt")

# Data must be in the following format:
'''
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    }
]
'''
with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/trans_result/src.txt","r",encoding="utf-8") as f:
    src = f.readlines()
f.close()

with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/trans_result/output_document_mdpo_repe","r",encoding="utf-8") as f:
    out = f.readlines()
f.close()

with open("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/trans_result/ref.txt","r",encoding="utf-8") as f:
    ref = f.readlines()
f.close()

data = []
for a,b,c in zip(src,out,ref):
    it = {
        "src": a.replace("\n", ""),
        "mt": b.replace("\n", ""),
        "ref": c.replace("\n", "")
    }
    data.append(it)
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)
print(model_output)