import sys
import os
import torch
import numpy as np
from transformers import DistilBertTokenizerFast
from training.train import AttributeExtractor

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from data.library.attribute_categories import ATTRIBUTE_CATEGORIES

class PreferenceAnalyzer:
    def __init__(self, model_path):
        self.model = AttributeExtractor().load_from_checkpoint(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.id2category = {i: cat for i, cat in enumerate(ATTRIBUTE_CATEGORIES.keys())}
        
    def analyze(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            category_probs, sentiment_probs = self.model(**inputs)
            
        return self._process_outputs(
            inputs.input_ids[0],
            category_probs[0],
            sentiment_probs[0]
        )
    
    def _process_outputs(self, input_ids, category_probs, sentiment_probs):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        category_preds = torch.argmax(category_probs, dim=-1)
        sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
        
        # Group consecutive tokens with same predictions
        current_span = []
        results = []
        
        for token, cat_idx, sent_idx in zip(tokens, category_preds, sentiment_preds):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            cat = self.id2category[cat_idx.item()]
            sentiment = "positive" if sent_idx.item() == 1 else "negative"
            
            if current_span and current_span[-1]["category"] == cat:
                current_span.append(token)
            else:
                if current_span:
                    results.append(self._format_span(current_span))
                current_span = [{
                    "token": token,
                    "category": cat,
                    "sentiment": sentiment
                }]
                
        return results
    
    def _format_span(self, span):
        return {
            "phrase": self.tokenizer.convert_tokens_to_string([s["token"] for s in span]),
            "category": span[0]["category"],
            "sentiment": span[0]["sentiment"],
            "confidence": np.mean([s["confidence"] for s in span])
        }