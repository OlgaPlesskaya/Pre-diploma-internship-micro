# utils/model_loader.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from streamlit import cache_resource

@cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model