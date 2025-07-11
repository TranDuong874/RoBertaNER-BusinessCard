from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
import json
from sklearn.model_selection import train_test_split
import yaml
import sys
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

# TODO:
# Add website label

# ========== UTILITIES ==========
def label_to_id(label_list):
    dic = {}
    for i, label in enumerate(label_list):
        dic[label] = i
    return dic
    
def id_to_label(label_list):
    dic = {}
    for i, label in enumerate(label_list):
        dic[i] = label
    return dic

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "eval_f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }
# ========== DATASET   ==========
class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_length):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.label2id = label2id
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data_set = json.load(f)
        
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        # ================================================================
        # SELECTED SAMPLE FORMAT
        # {
        #     tokens = ["Kính", "chào", "박", "지민"]
        #     ner_tags = ["O", "O", "B-Name", "I-Name"]
        # }
        # ================================================================
        # TOKENIZED SAMPLE
        # tokens: ['<s>', 'K', 'ĭ', 'nh', 'chào', '▁박', '▁지', '민', '</s>']
        # word_ids: [None, 0, 0, 0, 1, 2, 3, 3, None]
        # ================================================================
        # ALIGNED SAMPLE (Mapping word_ids -> labels)
        # labels = [-100, O, O, O, O, B-Name, I-Name, I-Name, -100]

        selected_example = self.data_set[idx]
        
        # Tokenize text
        tokenized_text = self.tokenizer(
            selected_example["tokens"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True
        )

        ner_token_labels = self._align_tokens(tokenized_text.word_ids(), selected_example)
        tokenized_text["labels"] = ner_token_labels
        return tokenized_text


    def _align_tokens(self, word_ids, example):
        """
            Convert character level tokens (e.g. "Ha": 1, "Noi": 2) to word level token (e.g. "Ha Noi" : B-Address)
        """
         # Token alignment
        ner_token_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Use -100 to ecode speical tokens [SEP], [CLS], [PAD]
                ner_token_labels.append(-100)  
            elif word_idx < len(example["ner_tags"]):
                if word_idx != previous_word_idx:
                    ner_token_labels.append(self.label2id.get(example["ner_tags"][word_idx], 0))
                else:
                    prev_label = example["ner_tags"][previous_word_idx]
                    if prev_label.startswith("B-"):
                        ner_token_labels.append(self.label2id.get("I-" + prev_label[2:], 0))
                    else:
                        ner_token_labels.append(self.label2id.get(prev_label, 0))
                previous_word_idx = word_idx
            else:
                # Ignore the token if word_idx exceed label list length
                ner_token_labels.append(-100)

        return ner_token_labels
        

class NERDataLoader(DataLoader):
    def __init__ (self, dataset, tokenizer, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.tokenizer = dataset.tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
    
    def collate_fn(self, batch):
        return self.data_collator(batch)

if __name__ == '__main__':
    # Get config path from command line
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'training_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['train']
    eval_cfg = config['eval']

    model_checkpoint = model_cfg["model_checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    label_list = data_cfg["label_list"]
    id2label_dict = id_to_label(label_list)
    label2id_dict = label_to_id(label_list)

    # Load the full dataset as a list
    with open(data_cfg['data_path'], 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # Split into train and test
    train_data, test_data = train_test_split(
        full_data,
        test_size=data_cfg.get('test_size', 0.2),
        random_state=data_cfg.get('random_state', 42)
    )

    # Save splits to temp files (or you can use in-memory datasets)
    with open(data_cfg['train_path'], 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(data_cfg['test_path'], 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    train_dataset = NERDataset(data_path=data_cfg['train_path'], tokenizer=tokenizer, label2id=label2id_dict, max_length=data_cfg.get('max_length', 128))
    test_dataset = NERDataset(data_path=data_cfg['test_path'], tokenizer=tokenizer, label2id=label2id_dict, max_length=data_cfg.get('max_length', 128))

    train_dataloader = NERDataLoader(train_dataset, tokenizer, batch_size=train_cfg["per_device_train_batch_size"], shuffle=True)
    test_dataloader = NERDataLoader(test_dataset, tokenizer, batch_size=eval_cfg["per_device_eval_batch_size"], shuffle=False)

    # ============== TRAINING ==============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, 
        num_labels=len(label_list), 
        id2label=id2label_dict, 
        label2id=label2id_dict,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir                      =train_cfg["output_dir"],
        gradient_checkpointing          =train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs   =train_cfg.get("gradient_checkpointing_kwargs", {}),
        save_strategy                   =train_cfg["save_strategy"],
        eval_strategy             =train_cfg["evaluation_strategy"],
        learning_rate                   =float(train_cfg["learning_rate"]),
        per_device_train_batch_size     =int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size      =int(eval_cfg["per_device_eval_batch_size"]),
        num_train_epochs                =float(train_cfg["num_train_epochs"]),
        weight_decay                    =float(train_cfg["weight_decay"]),
        logging_dir                     =train_cfg["logging_dir"],
        logging_steps                   =int(train_cfg["logging_steps"]),
        logging_strategy                =train_cfg["logging_strategy"],
        report_to                       =train_cfg["report_to"],
        fp16                            =train_cfg["fp16"],
        optim                           =train_cfg["optim"],                
        load_best_model_at_end          =train_cfg["load_best_model_at_end"],            
        metric_for_best_model           =train_cfg["metric_for_best_model"],
        greater_is_better               =train_cfg["greater_is_better"],
        save_total_limit                =int(train_cfg["save_total_limit"])
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=train_dataloader.data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()