from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
import json
import yaml
import sys
import numpy as np
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import pandas as pd
import os
import yaml

# HELPERS
from utils.data import NERDataset
from utils.tools import label_to_id, id_to_label
from utils.augmentations import SpaceLineAugmentation, ReplacementAugmentation
from utils.tools import make_compute_metrics, CSVLoggerCallback


LABEL_LIST = ["O", "B-Name", "I-Name", "B-Position", "I-Position", "B-Company", 
              "I-Company", "B-Address", "I-Address", "B-Phone", "I-Phone", "B-Email", 
              "I-Email", "B-Department", "I-Department", "B-Website", "I-Website"]
MODEL_CHECKPOINT = "Davlan/xlm-roberta-base-ner-hrl"
TRAIN_DATA_PATH = '/data/clean_ner_data.json'
VAL_DATA_PATH = '/pytess_ner_data.json'

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'training_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["checkpoint"])
    label2id = label_to_id(config["model"]["label_list"])
    id2label = id_to_label(config["model"]["label_list"])
    model = AutoModelForTokenClassification.from_pretrained(
        config["model"]["checkpoint"],
        num_labels=config["model"]["num_labels"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Dataset with augmentation
    augmentations = nn.Sequential(
        SpaceLineAugmentation(
            prob_space=config["dataset"]["augmentation"]["prob_space"],
            prob_newline=config["dataset"]["augmentation"]["prob_newline"],
            max_spaces=config["dataset"]["augmentation"]["max_spaces"]
        ),
        ReplacementAugmentation(
            prob=config["dataset"]["augmentation"]["replace_prob"]
        )
    )

    train_dataset = NERDataset(
        data_path=config["dataset"]["train_data_path"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=config["dataset"]["max_length"],
        augment=augmentations,  # Fixed from 'augment'
    )

    val_dataset = NERDataset(
        data_path=config["dataset"]["val_data_path"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=config["dataset"]["max_length"],
        augment=None,  # No augmentation for validation
    )


    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config["training"]["gradient_checkpointing_kwargs"],
        save_strategy=config["training"]["save_strategy"],          # "steps"
        eval_strategy=config["training"]["eval_strategy"],    # "steps"
        eval_steps=int(config["training"]["eval_steps"]) if "eval_steps" in config["training"] else None,
        
        # **Add save_steps since save_strategy is "steps" but config missing save_steps**
        save_steps=int(config["training"]["eval_steps"]) if "eval_steps" in config["training"] else 100, 
        
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["training"]["per_device_eval_batch_size"]),
        num_train_epochs=int(config["training"]["num_train_epochs"]),
        weight_decay=float(config["training"]["weight_decay"]),
        logging_dir=config["training"]["logging_dir"],
        logging_steps=int(config["training"]["logging_steps"]),
        logging_strategy=config["training"]["logging_strategy"],    # "steps"
        report_to=config["training"]["report_to"],
        fp16=config["training"]["fp16"],
        optim=config["training"]["optim"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        save_total_limit=int(config["training"]["save_total_limit"]),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "linear"),
        warmup_ratio=float(config["training"]["warmup_ratio"]) if "warmup_ratio" in config["training"] else None,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(config["model"]["label_list"]),
        callbacks=[CSVLoggerCallback(log_path=config["training"]["metrics_log_path"])]
    )

    # Train
    trainer.train()

    # Save the model
    trainer.save_model(f"{config['training']['output_dir']}/final_model")
    tokenizer.save_pretrained(f"{config['training']['output_dir']}/final_model")