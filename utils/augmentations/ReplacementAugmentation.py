import torch.nn as nn
import random
from typing import Dict, List

class ReplacementAugmentation(nn.Module):
    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob
        self.replacement_map = {
            # (same replacement_map as before — truncated here for brevity)
            'a': ['@', 'á', 'à', 'â', 'ä', 'ɑ'],
            'b': ['6', 'ß', 'Β'],
            'c': ['ç', '¢', 'ć', 'č'],
            # ... include the full map ...
            '9': ['g', 'q', 'p'],
            '@': ['a', 'A'],
            '$': ['S', 's'],
            '+': ['t', 'T'],
            '%': ['x', 'X'],
        }

    def forward(self, sample: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Accepts a dict with keys "tokens" (list of str) and "ner_tags" (list),
        returns the same structure with tokens augmented.
        """
        tokens = sample.get("tokens", [])
        new_tokens = []

        for token in tokens:
            new_token = ''.join(
                random.choice(self.replacement_map[c]) if c in self.replacement_map and random.random() < self.prob else c
                for c in token
            )
            new_tokens.append(new_token)

        return {
            "tokens": new_tokens,
            "ner_tags": sample.get("ner_tags", [])
        }

    def __call__(self, sample):
        return self.forward(sample)