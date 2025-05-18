import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection, F1Score, Precision, Recall
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
from transformers import DistilBertTokenizerFast
from training.train import AttributeExtractor

from models.architecture.model import AttributeExtractors
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data.library.attribute_categories import ATTRIBUTE_CATEGORIES

class TrainingModule(pl.LightningModule):
    def __init__(self, model, class_weights=None):
        super().__init__()
        self.model = model
        self.class_weights = class_weights
        
        # Loss functions with class weighting
        self.category_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100
        )
        self.sentiment_loss = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def _shared_step(self, batch, stage):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Split labels
        category_labels = labels // 2
        sentiment_labels = labels % 2
        
        # Forward pass
        category_logits, sentiment_logits = self(input_ids, attention_mask)
        
        # Calculate losses
        cat_loss = self._masked_loss(category_logits, category_labels, 
                                   self.category_loss, attention_mask)
        sent_loss = self._masked_loss(sentiment_logits, sentiment_labels,
                                    self.sentiment_loss, attention_mask)
        total_loss = cat_loss + sent_loss
        
        # Log metrics
        self.log_dict({
            f"{stage}_cat_loss": cat_loss,
            f"{stage}_sent_loss": sent_loss,
            f"{stage}_total_loss": total_loss
        }, prog_bar=True)
        
        return total_loss
    
    def _masked_loss(self, logits, labels, loss_fn, mask):
        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1, logits.size(-1))[active_loss]
        active_labels = labels.view(-1)[active_loss]
        return loss_fn(active_logits, active_labels)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            category_logits, sentiment_logits = self(
                batch["input_ids"], 
                batch["attention_mask"]
            )
        return {
            "category_probs": torch.softmax(category_logits, dim=-1),
            "sentiment_probs": torch.softmax(sentiment_logits, dim=-1)
        }
    



def calculate_class_weights(dataset):
    """Compute balanced class weights for imbalanced data"""
    all_labels = np.concatenate([sample["labels"] for sample in dataset])
    category_labels = (all_labels // 2).astype(int)
    
    counts = np.bincount(category_labels[category_labels != -100])
    weights = 1. / (counts + 1e-6)
    return torch.tensor(weights / weights.sum(), dtype=torch.float32)

# Initialize components
dataset = load_from_disk("data/synthetic_dataset")
train_data, val_data = dataset.train_test_split(test_size=0.2).values()

model = AttributeExtractor()
class_weights = calculate_class_weights(train_data)
train_module = TrainingModule(model, class_weights=class_weights)

# Configure Trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,
    precision=16,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    enable_progress_bar=True
)

# Execute training
trainer.fit(
    train_module,
    DataLoader(train_data, batch_size=32, shuffle=True),
    DataLoader(val_data, batch_size=64) )



class AttributeMetrics(pl.LightningModule):
    def __init__(self, num_categories):
        super().__init__()
        metrics = MetricCollection({
            "cat_f1": F1Score(num_classes=num_categories, average="macro"),
            "cat_precision": Precision(num_classes=num_categories, average="macro"),
            "cat_recall": Recall(num_classes=num_categories, average="macro")
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        
    def update_metrics(self, preds, labels, mask, stage):
        active_preds = preds[mask == 1]
        active_labels = labels[mask == 1]
        
        if stage == "train":
            self.train_metrics.update(active_preds, active_labels)
        else:
            self.val_metrics.update(active_preds, active_labels)
            
    def log_metrics(self, stage):
        if stage == "train":
            self.log_dict(self.train_metrics.compute(), prog_bar=True)
            self.train_metrics.reset()
        else:
            self.log_dict(self.val_metrics.compute(), prog_bar=True)
            self.val_metrics.reset()



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