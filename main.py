from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification
# TODO:
# Add website labe

# ========== UTILITIES ==========
def label2id(label, label_map):

def label2id(label, label_map):

# ========== DATASET   ==========
class NERDataset(Dataset):
    def __init__(self, json_file, tokenizer, label_map):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.label_map = label_map
        
        # Tokenize data
        # Add a TQDM loader

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        pass

class NERDataLoader(DataLoader):
    def __init__ (self, dataset, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.tokenizer = dataset.tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
    
    def collate_fn(self, batch):
        return self.data_collator(batch)

# ========== MODEL   ==========
class NERModel(torch.nn.Module):
    def __init__(self, model):
        super(NERModel, self).__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask=None):
        pass

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
    model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
    