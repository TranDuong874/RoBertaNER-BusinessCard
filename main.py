# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")