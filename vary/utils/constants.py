CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


ROOT_PATH = '/data/public/ucaswei/data/'

CONVERSATION_DATA = {

    # pair 4m
    'laion-coco-4m': {
        'images': '',
        'annotations': '',
    }, 

    'cc665k': {
        'images': "/path_to/LLaVA1.5/images/",
        'annotations': "/path_to/LLaVA1.5/llava_v1_5_66k.json",
    },
    'pdf_trans':{
        'images':"/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/",
        'annotations':"/home/notebook/data/personal/S9055503/pdf_zh.json"
    },
    'cot_pdf_trans':{
        'images':"/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/",
        'annotations':"/home/notebook/data/personal/S9055503/cot_pdf_zh.json"
    },
    'pdf_en': {
        'images': "/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/",
        'annotations': "/home/notebook/data/personal/S9055503/clear_en_data_30w.json",
    },
    'synthdog':{
        'images':"/home/notebook/data/personal/S9055503/",
        'annotations':"/home/notebook/data/personal/S9055503/synthdog.json"
    },
    'wmt_trans':{
        'images':None,
        'annotations':"/home/notebook/data/personal/S9055503/multi_prompt/multi_only_trans.json"
    },
    'multi_ocr':{
        'images':"/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/",
        'annotations':"/home/notebook/data/personal/S9055503/multi_prompt/multi_prompt_ocr.json"
    },
    'multi_trans': {
        'images': "/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/",
        'annotations': "/home/notebook/data/personal/S9055503/multi_prompt/multi_prompt_trans.json",
    },
    'text_vqa':{
        'images':"/home/notebook/data/personal/S9055503/textvqa/train_images/",
        'annotations':"/home/notebook/data/personal/S9055503/textvqa/train_textvqa.json"
    },
    'ocr_vqa':{
        'images':'/home/notebook/data/personal/S9055503/ocrvqa/images/',
        'annotations':'/home/notebook/data/personal/S9055503/ocrvqa/train_ocrvqa.json'
    },
    'wiki_trans':{
        'images':None,
        'annotations':"/home/notebook/data/personal/S9055503/multi_prompt/wiki.json"
    },
    'dpo_train':{
        'images':"/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/",
        'annotations':"/home/notebook/data/personal/S9055503/dpo_data/dpo_train_data.json"
    },
    'mdpo_train':{
        'images':"/home/notebook/data/personal/S9055503/data/pdf_data/pdf_en_30w/",
        'annotations':"/home/notebook/data/personal/S9055503/mdpo_train_data.json"
    },
    'docvqa_train': {
        'images': "",
        'annotations': "",
    },

    'chartqa_train': {
        'images': "",
        'annotations': "",
    },



}