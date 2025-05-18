import sys
import os
from transformers import DistilBertTokenizerFast
import numpy as np
import torch


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from library.attribute_categories import ATTRIBUTE_CATEGORIES


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def align_labels(sample):
    """Convert annotations to token-level labels"""
    tokens = tokenizer(sample["text"], truncation=True, padding="max_length", max_length=128)
    labels = np.zeros(len(tokens["input_ids"]), dtype=int)
    
    # Convert character positions to token positions
    for ann in sample["annotations"]:
        # Find token positions for each annotation phrase
        char_start = sample["text"].find(ann["phrase"])
        char_end = char_start + len(ann["phrase"])
        
        # Convert to token positions
        token_start = tokens.char_to_token(char_start)
        token_end = tokens.char_to_token(char_end - 1)
        
        if token_start and token_end:
            # Encode category + sentiment: [CATEGORY_ID][SENTIMENT_FLAG]
            cat_id = list(ATTRIBUTE_CATEGORIES.keys()).index(ann["category"])
            sentiment_flag = 1 if ann["sentiment"] == "positive" else 0
            labels[token_start:token_end+1] = cat_id * 2 + sentiment_flag
    
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels.tolist()
    }

def dataset_to_dataloader(dataset, batch_size=16):
    """Convert HF dataset to torch DataLoader"""
    dataset = dataset.map(align_labels, batched=False)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)