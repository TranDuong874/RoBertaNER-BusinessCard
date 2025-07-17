import json
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_length=128, augment=None):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.label2id = label2id
        self.max_length = max_length
        self.augment = augment

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
        
        if self.augment:
            selected_example = self.augment(selected_example)
            
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
        