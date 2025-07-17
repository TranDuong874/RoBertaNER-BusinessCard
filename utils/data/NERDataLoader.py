from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader, Dataset

class NERDataLoader(DataLoader):
    def __init__ (self, dataset, tokenizer, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.tokenizer = dataset.tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
    
    def collate_fn(self, batch):
        return self.data_collator(batch)
