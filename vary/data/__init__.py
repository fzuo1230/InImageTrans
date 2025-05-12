
import torch
import transformers
from dataclasses import dataclass, field

from vary.utils.constants import *


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        images = [torch.stack(instance['image']) for instance in instances]


        images_high = [torch.stack(instance['image_high']) for instance in instances]

        images = list(zip(images, images_high))


        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )
        #print(batch)
        return batch

@dataclass
class DataCollatorForDpoDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):

        input_ids, labels, input_ids_reject,rejecteds = tuple([instance[key] for instance in instances] for key in ("chosen_input_ids", "chosen_labels","rejected_input_ids","rejected_labels"))
        images = [torch.stack(instance['image']) for instance in instances]
        #print(images)
        #print(type(images))

        images_high = [torch.stack(instance['image_high']) for instance in instances]

        images = list(zip(images, images_high))
        

        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_reject,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        rejected_labels = torch.nn.utils.rnn.pad_sequence(
            rejecteds,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
            rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )
        #print(batch)
        return batch

@dataclass
class DataCollatorFormDpoDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):

        input_ids, labels, input_ids_reject,rejecteds = tuple([instance[key] for instance in instances] for key in ("chosen_input_ids", "chosen_labels","rejected_input_ids","rejected_labels"))
        images = [torch.stack(instance['image']) for instance in instances]
        #print(images)
        #print(type(images))

        images_high = [torch.stack(instance['image_high']) for instance in instances]

        images = list(zip(images, images_high))

        rejected_images = [torch.stack(instance['rejected_image']) for instance in instances]

        rejected_images_high = [torch.stack(instance['rejected_image_high']) for instance in instances]

        rejected_images = list(zip(rejected_images,rejected_images_high))

        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_reject,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        rejected_labels = torch.nn.utils.rnn.pad_sequence(
            rejecteds,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
            rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
            rejected_images=rejected_images,
        )
        #print(batch)
        return batch


def make_supervised_data_module(interleave, with_box, tokenizer, data_args):

    if data_args.conversation_version == 'mpt':
        from vary.data.conversation_dataset_qwen import ConversationDataset
        dataset_cls = ConversationDataset
    elif data_args.conversation_version == 'opt':
        from vary.data.caption_opt import CaptionDataset
        dataset_cls = CaptionDataset

    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        datasets=data_args.datasets,
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high = data_args.image_processor_high,
            box_limit=data_args.box_limit,
        )
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_dpo_data_module(interleave, with_box, tokenizer, data_args):

    if data_args.conversation_version == 'mpt':
        from vary.data.conversation_dataset_qwen import ConversationDataset,DpoDataset
        dataset_cls = DpoDataset
    elif data_args.conversation_version == 'opt':
        from vary.data.caption_opt import CaptionDataset
        dataset_cls = CaptionDataset

    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        datasets=data_args.datasets,
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high = data_args.image_processor_high,
            box_limit=data_args.box_limit,
        )
    )
    data_collator = DataCollatorForDpoDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_mdpo_data_module(interleave, with_box, tokenizer, data_args):

    if data_args.conversation_version == 'mpt':
        from vary.data.conversation_dataset_qwen import ConversationDataset,DpoDataset
        dataset_cls = DpoDataset
    elif data_args.conversation_version == 'opt':
        from vary.data.caption_opt import CaptionDataset
        dataset_cls = CaptionDataset

    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        datasets=data_args.datasets,
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high = data_args.image_processor_high,
            box_limit=data_args.box_limit,
        )
    )
    data_collator = DataCollatorFormDpoDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

