import torch.nn as nn
import numpy as np

class SpaceLineAugmentation(nn.Module):
    def __init__(self, prob_space=0.5, prob_newline=0.5, max_spaces=4):
        super().__init__()
        self.prob_space = prob_space
        self.prob_newline = prob_newline
        self.max_spaces = max_spaces
    
    def augment_sample(self, sample, prob_space=0.5, prob_newline=0.5, max_spaces=4):
        """
            Augment a single data sample by adding space and new line

            Params:
                sample: a single training sample with the following format
                    {
                        "tokens": ["raw", "string", "tokens"],
                        "ner_tags": ["O", "O", "O"]
                    }
                prob_space: Probability of adding space characters
                prob_newline: Probability of adding a newline
        """
        
        tokens = sample['tokens']
        ner_tags = sample['ner_tags']
        new_tokens = []
        new_tags = []
        for i, token in enumerate(tokens):
            new_tokens.append(token)
            new_tags.append(ner_tags[i])
            if i < len(tokens) - 1:
                r = np.random.rand()
                if r < prob_newline:
                    new_tokens.append('\n')
                    new_tags.append('O')
                elif r < prob_newline + prob_space:
                    num_spaces = np.random.randint(1, max_spaces + 1)
                    new_tokens.append(' ' * num_spaces)
                    new_tags.append('O')
        return {'tokens': new_tokens, 'ner_tags': new_tags}

    def forward(self, x):
        return self.augment_sample(x, self.prob_sapce, self.prob_newline, self.max_spaces)

    def __call__(self, tokens, labels=None):
        """Allows the module to be called like a function."""
        return self.forward(tokens, labels)