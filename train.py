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
from utils.augmentations import SpaceLineAugmentation
from utils.tools import make_compute_metrics, CSVLoggerCallback


LABEL_LIST = ["O", "B-Name", "I-Name", "B-Position", "I-Position", "B-Company", 
              "I-Company", "B-Address", "I-Address", "B-Phone", "I-Phone", "B-Email", 
              "I-Email", "B-Department", "I-Department", "B-Website", "I-Website"]
MODEL_CHECKPOINT = "Davlan/xlm-roberta-base-ner-hrl"
TRAIN_DATA_PATH = '/data/clean_ner_data.json'
VAL_DATA_PATH = '/pytess_ner_data.json'

# Refactored inference function
def run_inference(model, tokenizer, dataset=None, single_input=None, id2label=None, output_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    model.to(device)
    results = []

    def group_predictions_to_json(tokens, predictions, word_ids):
        categories = {
            'Name': [], 'Company': [], 'Position': [], 'Department': [],
            'Address': [], 'Phone': [], 'Email': [], 'Website': [], 'Other': []
        }
        current_category = None
        current_value = []
        word_predictions = []

        # Align subword predictions to word-level predictions
        previous_word_id = None
        for pred, word_id in zip(predictions, word_ids):
            if word_id is None:  # Skip special tokens (e.g., [CLS], [SEP])
                continue
            if word_id != previous_word_id:  # New word
                word_predictions.append(pred)
                previous_word_id = word_id

        # Group predictions into entities
        for word, label_id in zip(tokens, word_predictions):
            label = id2label[label_id]
            if label == "O":
                if current_category:
                    categories[current_category].append(' '.join(current_value))
                    current_category = None
                    current_value = []
                categories["Other"].append(word)
                continue

            prefix, entity = label.split("-", 1)
            if prefix == "B" or (prefix == "I" and current_category != entity):
                if current_category:
                    categories[current_category].append(' '.join(current_value))
                current_category = entity
                current_value = [word]
            elif prefix == "I" and current_category == entity:
                current_value.append(word)

        # Add final entity
        if current_category and current_value:
            categories[current_category].append(' '.join(current_value))

        # Simplify single-element lists
        for key in categories:
            if len(categories[key]) == 1:
                categories[key] = categories[key][0]
            elif not categories[key]:
                categories[key] = ""

        return categories

    if single_input:
        # Single input inference
        tokenized_inputs = tokenizer(
            single_input,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        word_ids = tokenized_inputs.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = model(**tokenized_inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"].squeeze().tolist())

        result = group_predictions_to_json(single_input, predictions, word_ids)
        results.append(result)

    elif dataset:
        # Dataset inference
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=data_collator
        )
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                word_ids = [tokenizer.convert_ids_to_tokens(batch["input_ids"][i], skip_special_tokens=True) for i in range(len(batch["input_ids"]))]
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()

                for i in range(len(batch["input_ids"])):
                    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i], skip_special_tokens=True)
                    batch_word_ids = tokenizer.word_ids(batch_index=i)
                    result = group_predictions_to_json(tokens, predictions[i], batch_word_ids)
                    results.append(result)

    # Save results to JSON
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Inference results saved to {output_path}")

    return results

if __name__ == "__main__":
    # Load configuration
    with open("training_config.yaml", "r") as f:
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config["training"]["gradient_checkpointing_kwargs"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy=config["training"]["eval_strategy"],
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir=config["training"]["logging_dir"],
        logging_steps=config["training"]["logging_steps"],
        logging_strategy=config["training"]["logging_strategy"],
        report_to=config["training"]["report_to"],
        fp16=config["training"]["fp16"],
        optim=config["training"]["optim"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        save_total_limit=config["training"]["save_total_limit"]
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

    if config["inference"]["run_inference"]:
        inference_dataset = NERDataset(
            data_path=config["inference"].get("inference_data_path", config["dataset"]["val_data_path"]),
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=config["dataset"]["max_length"],
            transform=None,
            is_train=False
        )

        results = run_inference(
            model=model,
            tokenizer=tokenizer,
            dataset=inference_dataset,
            single_input=None,  # Set to single_input for single-input inference
            id2label=id2label,
            output_path=config["inference"]["output_path"]
        )