# utils/data_processor.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .model_loader import load_model_and_tokenizer
from config.settings import MODEL_PATH, MAX_LENGTH, SCALING_FACTOR
from .api_client import get_categories, get_subcategories


class PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str):
            text = str(text)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def predict_labels(texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
    model.to(device)
    model.eval()

    dataset = PredictionDataset(texts=texts, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_active_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs_batch = torch.sigmoid(outputs.logits).cpu().numpy()
            probs_batch = probs_batch.squeeze(1) if probs_batch.ndim == 3 else probs_batch

            adaptive_thresholds = np.max(probs_batch, axis=1) * SCALING_FACTOR
            active_labels_batch = (probs_batch > adaptive_thresholds[:, np.newaxis]).astype(int)

            active_indices_batch = [
                ",".join(map(str, (np.where(active_labels)[0] + 1)))
                if np.any(active_labels) else ""
                for active_labels in active_labels_batch
            ]

        all_active_labels.extend(active_indices_batch)

    return all_active_labels

def build_label_dict():
    label_dict = {}
    
    categories = get_categories()
    for category in categories:
        category_id = category["identifier"]
        subcategories = get_subcategories(category_id)
        
        for sub in subcategories:
            label_dict[sub["identifier"]] = sub["name"]
    
    return label_dict

def process_uploaded_file(df):
    texts = df.iloc[:, 0].values
    predicted_indices = predict_labels(texts)

    output_df = pd.DataFrame({"text": texts, "predicted_labels": predicted_indices})
    #labels_df = pd.read_excel("Классификации.xlsx")
    #label_dict = dict(zip(labels_df['id'], labels_df['level_3']))

    label_dict = build_label_dict()

    def map_labels(label_str, label_map):
        try:
            ids = [int(x.strip()) for x in str(label_str).split(",")]
            return "; ".join([label_map[id] for id in ids if id in label_map])
        except Exception as e:
            return ""

    output_df['predicted_labels'] = output_df['predicted_labels'].apply(lambda x: map_labels(x, label_dict))
    output_df = output_df[['text', 'predicted_labels']]
    return output_df