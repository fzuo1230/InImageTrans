# InImageTrans：Multimodal LLM-based Text Image Machine Translation

This is the official repository of InImageTrans. Our work mainly conducts end-to-end image-text translation tasks based on Multimodal LLM.

# Create Environment

conda create -n InImageTrans python=3.10


pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118


pip install -r requirements.txt

# SFT Training

First, you need to configure your dataset path in ./utils/constants.py and name your dataset. Then, you need to change the hyperparameters in train.sh, mainly including the dataset name, pre-trained model path, deepspeed_config, etc. After the modification, you can directly run the following command line to start training:

bash train.sh


