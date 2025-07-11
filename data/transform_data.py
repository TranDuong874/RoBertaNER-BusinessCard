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
    
if __name__ == '__main__':
    ner_json = read_json('data_ner.json')

    # Update B-Website tags
    updated_ner_data = update_website_tag(ner_json)

    write_json('data_ner_updated.json', updated_ner_data)

