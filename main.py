from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        pass
    
class NERDataLoader(DataLoader):
    def __init__ (self, dataset, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def collate_fn(self, batch):
        pass

class NERModel(torch.nn.Module):
    def __init__(self, model):
        super(NERModel, self).__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask=None):
        pass

if __name__ == '__main__':
    pass