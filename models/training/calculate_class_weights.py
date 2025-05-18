import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import torch
import pytorch_lightning as pl
from train import TrainingModule
from architecture.model import AttributeExtractor


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
    DataLoader(val_data, batch_size=64)
)