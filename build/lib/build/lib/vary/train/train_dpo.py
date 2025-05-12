import sys
sys.path.append("/home/notebook/code/personal/S9055503/Vary-main/Vary-master/mt_trl")
from my_trl.trainer import DPOConfig
import logging
import pathlib
import torch
from transformers import AutoTokenizer
import transformers
import os
import copy
from torch.utils.data import Dataset

import random
import json
from PIL import Image
from vary.train.trainer_vit_fixlr import varyTrainer,varyDpoTrainer
from vary.model import *
from vary.data import make_supervised_data_module,make_dpo_data_module,make_mdpo_data_module
from vary.utils.arguments import *
from vary.utils.utils import smart_tokenizer_and_embedding_resize
from vary.model.vision_encoder.sam import build_sam_vit_b
from vary.utils.utils import disable_torch_init

from trl.trainer.utils import DPODataCollatorWithPadding

from datasets import Dataset

def make_conv(prompt,answer):
    return [
        {
            "from":"human",
            "value":prompt,
        },
        {
            "from":"gpt",
            "value":answer,
        },
    ]

def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def load_data(data_args):
    if 'jsonl' in data_args.datasets:
        data_list = load_jsonl(data_args.datasets)
    else: 
        data_list = load_json(data_args.datasets)
    return data_list
'''
class DPODataset(Dataset):

    def __init__(self,data_path:str,
                 tokenizer:transformers.PreTrainedTokenizer,
                 data_args:DataArguments):
        super(Dataset, self).__init__()
        list_data_dict = load_data(data_args)
        #if data_args.num_sample is not None:
        #   list_data_dict = list_data_dict[:data_args.num_sample]
        print(list_data_dict[0])
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.training_modal = data_args.training_modal

    def __len__(self):
        return len(self.list_data_dict)
        
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 256
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            length_list.append(cur_len)
        return length_list

    def __getitem__(self,i) -> Dict[str, torch.Tensor]:
        try:
            has_X = None

            data_dict = copy.deepcopy(self.list_data_dict[i])
            #print(data_dict)
            if self.training_modal == 'image':
                image_file = data_dict['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor
                image_processor_high = self.data_args.image_processor_high

                image = Image.open(os.path.join(image_folder,image_file)).convert("RGB")
                image_high = image.copy()

                if self.data_args.image_aspect_ratio == 'square':
                    image = image_processor.preprocess(image,return_tensors='pt')['pixel_values'][0]
                
                image_high = image_processor_high.preprocess(image_high)

                prompt = data_dict['prompt']
                prompt = prompt.replace("<image>","").strip()
                prompt = "<image>\n" + prompt
                data_dict["prompt"] = prompt
                has_X = "image"
            
            else:
                raise("Training modal not supported!")
            
            data_dict['has_X'] = has_X
            if has_X == 'image':
                data_dict['image'] = image
                data_dict['image_high'] = image_high
            
            return data_dict

        except Exception as e:
            print(f'Error with {e}, {self.list_data_dict[i]}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                          data_args) -> Dict:
    train_dataset = DPODataset(tokenizer=tokenizer,
                               data_args=data_args,
                               data_path=data_args.datasets)
    return train_dataset
'''
'''
@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    def collate(self,batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue

                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad,batch_first=True,padding_value=padding_value)

            else:
                padded_batch[k] = [ex[k] for ex in batch]
            
        for k in ['chosen_input_ids','rejected_input_ids']:
            attn_k = k.replace('input_ids', 'attention_mask')
            padded_batch[attn_k] = padded_batch[k].ne(self.tokenizer.pad_token_id)
        return padded_batch

    def tokenize_batch_element(
        self,
        prompt:str,
        chosen:str,
        rejected:str,
        has_X:str = None
    ) -> Dict:

        batch = {}

        chosen_sources = make_conv(prompt, chosen)
        rejected_sources = make_conv(prompt, rejected)
        chosen_data_dict =
''' 
def train():
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    #print(training_args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, padding_side="right", model_max_length=training_args.model_max_length,)
    
    #data_args.training_modal = "image"

    model = varyQwenForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
 
    #model_ref = varyQwenForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token='<|endoftext|>'), 
        tokenizer=tokenizer,
        model=model)
    '''
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token='<|endoftext|>'), 
        tokenizer=tokenizer,
        model=model_ref)
    '''

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
 
    vision_tower_dict = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        freeze_vision_tower=model_args.freeze_vision_tower,
        use_im_start_end=model_args.use_im_start_end,
        vision_select_layer=model_args.vision_select_layer,
        dtype=dtype,
        device=training_args.device
    )

    model.initialize_vision_tokenizer(
        tokenizer=tokenizer,
        freeze_lm_model=model_args.freeze_lm_model,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )

    model.to(dtype=dtype,device=training_args.device)

    data_args.image_token_len = 256
    data_args.image_processor = vision_tower_dict['image_processor']
    data_args.image_processor_high = vision_tower_dict['image_processor_high']
    data_args.use_im_start_end = model_args.use_im_start_end
    #data_args.image_folder = "/home/notebook/data/personal/S9055503/test_image/"

    if model_args.freeze_lm_model:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector_vary.parameters():
            p.requires_grad = True
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True

        if not model_args.freeze_vision_tower:
            model.get_model().vision_tower.requires_grad_(True)
            model.get_model().vision_tower_high.requires_grad_(True)
        
    params_grad = [p.numel() for n,p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

    #print(data_args)
    train_dataset = make_mdpo_data_module(tokenizer=tokenizer, 
                                         data_args=data_args,
                                         interleave=training_args.interleave,
                                         with_box=training_args.with_box)
    #print(train_dataset["train_dataset"][0])

    dpo_train_dataset = {
        'images':[
            [Image.open("/home/notebook/data/personal/S9055503/test_image/test1.png")],
            [Image.open("/home/notebook/data/personal/S9055503/test_image/test2.png")],
        ],
        'prompt':[
            '<image>\ntranslate the OCR results into Chinese:',
            '<image>\ntranslate the OCR results into Chinese:',
        ],
        'chosen':[
            '我喜欢你',
            '我喜欢你',
        ],
        'rejected':[
            '我喜欢你我喜欢你我喜欢你我喜欢你我喜欢你我喜欢你我喜欢你我喜欢你',
            '我喜欢你我喜欢你我喜欢你我喜欢你我喜欢你',
        ]
    }
    '''
    training_args.max_length = training_args.model_max_length
    training_args.max_prompt_length = 512
    training_args. max_target_length = training_args.model_max_length
    training_args.model_init_kwargs = None
    ref_model_init_kwargs.ref_model_init_kwargs = None
    '''
    #print(training_args)
    dpo_args = DPOConfig(
        beta=0.1,
        output_dir=training_args.output_dir,
        bf16=training_args.bf16,
        dataloader_num_workers=training_args.dataloader_num_workers,
        deepspeed=training_args.deepspeed,
        gradient_checkpointing=training_args.gradient_checkpointing,
        learning_rate=training_args.learning_rate,
        logging_steps=training_args.logging_steps,
        max_length=training_args.model_max_length,
        max_prompt_length=512,
        max_target_length=training_args.model_max_length,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        report_to=training_args.report_to,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        tf32=training_args.tf32,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        remove_unused_columns=False,
        )
    
    #print(dpo_args)
    #print(dpo_args)
    dpo_trainer = varyDpoTrainer(
        model=model,
        ref_model=None,
        #args=dpo_args,
        args=dpo_args,
        #beta=0.1,
        tokenizer=tokenizer,
        **train_dataset
    )

    #dpo_trainer.train()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        dpo_trainer.train(resume_from_checkpoint=True)
    else:
        #check = torch.load("/home/notebook/data/personal/S9055503/pretrained_model")
        #model.load_state_dict(check['model_state_dict'])
        dpo_trainer.train()
    dpo_trainer.save_state()
    dpo_trainer._safe_save(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
        