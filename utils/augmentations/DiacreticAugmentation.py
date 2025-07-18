import torch.nn as nn
import unicodedata
import random
from typing import Dict, List

class DiacreticAugmentation(nn.Module):
    def __init__(self, prob: float = 0.3):
        """
        Args:
            prob (float): Probability of removing diacritics per token.
        """
        super().__init__()
        self.prob = prob

    def remove_diacritics(self, text: str) -> str:
        # Normalize and remove combining marks (diacritics)
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in nfkd if not unicodedata.combining(c)])

    def forward(self, sample: Dict[str, List[str]]) -> Dict[str, List[str]]:
        tokens = sample.get("tokens", [])
        new_tokens = []

        for token in tokens:
            if random.random() < self.prob:
                new_tokens.append(self.remove_diacritics(token))
            else:
                new_tokens.append(token)

        return {
            "tokens": new_tokens,
            "ner_tags": sample.get("ner_tags", [])
        }
