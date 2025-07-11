import re
import json

def update_website_tag(data):
    website_pattern = re.compile(r'^(https?://\S+|www\.\S+)$')

    for item in data:
        tokens = item['tokens']
        ner_tags = item['ner_tags']

        for i, token in enumerate(tokens):
            if website_pattern.match(token):
                ner_tags[i] = 'B-Website'
    
    return data

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        ner_json = json.load(file)
    return ner_json

def write_json(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    
import json
import random

def augment_tokens(tokens, prob_space=0.3, prob_newline=0.1, max_spaces=10):
    """Randomly add spaces (of variable length) and newlines between tokens."""
    new_tokens = []
    for i, token in enumerate(tokens):
        new_tokens.append(token)
        if i < len(tokens) - 1:
            r = random.random()
            if r < prob_newline:
                new_tokens.append('\n')
            elif r < prob_newline + prob_space:
                num_spaces = random.randint(1, max_spaces)
                new_tokens.append(' ' * num_spaces)
    return new_tokens

def augment_data(input_path, output_path, prob_space=0.9, prob_newline=0.5, max_spaces=10):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    augmented = []
    for sample in data:
        tokens = sample['tokens']
        ner_tags = sample['ner_tags']
        new_tokens = []
        new_tags = []
        for i, token in enumerate(tokens):
            new_tokens.append(token)
            new_tags.append(ner_tags[i])
            if i < len(tokens) - 1:
                r = random.random()
                if r < prob_newline:
                    new_tokens.append('\n')
                    new_tags.append('O')
                elif r < prob_newline + prob_space:
                    num_spaces = random.randint(1, max_spaces)
                    new_tokens.append(' ' * num_spaces)
                    new_tags.append('O')
        augmented.append({'tokens': new_tokens, 'ner_tags': new_tags})
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    ner_json = read_json('data_ner.json')

    # Update B-Website tags
    updated_ner_data = update_website_tag(ner_json)

    write_json('data_ner_updated.json', updated_ner_data)

    # Example usage:
    augment_data('data_ner_updated.json', 'data_ner_augmented.json', max_spaces=10)