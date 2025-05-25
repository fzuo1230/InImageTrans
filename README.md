# InImageTrans：Multimodal LLM-based Text Image Machine Translation

This is the official repository of InImageTrans. Our work mainly conducts end-to-end image-text translation tasks based on Multimodal LLM.

# MCiT benchmark

Due to the large amount of our benchmark data that cannot be directly uploaded to GitHub, we have shared a cloud storage link for download. The link to the cloud storage is as follows:

https://pan.baidu.com/s/136AQMNWCJOzMKXenIexT8A?pwd=3991

# Create Environment

conda create -n InImageTrans python=3.10


pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118


pip install -r requirements.txt

# SFT Training

First, you need to configure your dataset path in ./utils/constants.py and name your dataset. Then, you need to change the hyperparameters in train.sh, mainly including the dataset name, pre-trained model path, deepspeed_config, etc. After the modification, you can directly run the following command line to start training:

bash train.sh

# DPO Training

DPO training is different from SFT training because we modified it in mt_trl based on huggingface's trl library.

1.If you have huggingface's trl library installed, please uninstall it.

2.Different from the organization of SFT data, DPO training data adds reject_label and reject_image on the basis of SFT data. Please adjust your dataset format accordingly.

3.Similar to SFT Training, you need to adjust the corresponding hyperparameters in train_mdpo.sh, and then run the following command for training:

bash train_mdpo.sh

# Reference

You need to set the hyperparameters in infer.sh, including the model path and image path, and then execute the following command:

bash infer.sh

