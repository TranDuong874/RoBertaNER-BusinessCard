from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
import json
import yaml
import sys
import numpy as np
import torch.nn as nn

# HELPERS
from utils.data import NERDataLoader, NERDataset
from utils.tools import label_to_id, id_to_label
from utils.augmentations import SpaceLineAugmentation

LABEL_LIST = ["O", "B-Name", "I-Name", "B-Position", "I-Position", "B-Company", 
              "I-Company", "B-Address", "I-Address", "B-Phone", "I-Phone", "B-Email", 
              "I-Email", "B-Department", "I-Department", "B-Website", "I-Website"]

MODEL_CHECKPOINT = "Davlan/xlm-roberta-base-ner-hrl"
DATA_PATH = '/data/clean_ner_data.json'

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    augmentations = nn.Sequential(
        SpaceLineAugmentation(prob_space=0.8, prob_newline=0.5, max_space=10)
    )

    dataset = NERDataset(
        data_path=DATA_PATH, 
        tokenizer=tokenizer, 
        label2id=label_to_id(LABEL_LIST), 
        max_length=128,
        transform=augmentations)

    dataloader = NERDataLoader(dataset, tokenizer, batch_size=32, shuffle=True)
