# data_loaders.py
import logging
from typing import Dict, List, Tuple, Optional, Union
import os
import sys

# Adjust parent directory in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import DistilBertTokenizerFast
import nlpaug.augmenter.word as naw

from data.processing.feature_engineer import FeatureEngineer
from data.processing.tokenization import ClothingTokenizer
from data.generator.augmenter import DataAugmenter

logger = logging.getLogger(__name__)

class PreferenceDataset(TorchDataset):
    """PyTorch Dataset for clothing preference data"""
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: ClothingTokenizer,
        max_length: int = 128,
        augment: bool = True,
        aug_p: float = 0.3
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmenter = DataAugmenter(aug_p=aug_p) if augment else None
        # Initialize the feature engineer once
        self.feature_engineer = FeatureEngineer()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        original_text = sample.get("text", "")
        
        # Apply augmentation if enabled
        if self.augment and self.augmenter:
            sample = self._augment_sample(sample, original_text)
        
        # Extract features and add them to the sample if needed
        try:
            features = self.feature_engineer.extract_features(sample["text"])
            sample["features"] = features  # You can later use these features in training if desired
            logger.debug("Feature extraction successful.")
        except Exception as e:
            logger.error(f"Feature extraction failed at idx {idx}: {str(e)}")
        
        # Tokenize text (the tokenization function is assumed to handle cleaning)
        encoding = self.tokenizer.tokenize_with_cleaning(
            sample["text"],
            max_length=self.max_length
        )
        
        # Convert encoding to PyTorch tensors
        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        }
        
        # Add labels if available from tokenization or the sample
        if "labels" in sample:
            item["labels"] = torch.tensor(encoding.get("labels", sample["labels"]), dtype=torch.long)
        
        return item

    def _augment_sample(self, sample: Dict, original_text: str) -> Dict:
        """Apply lightweight augmentation during training. Ensure critical features are retained."""
        try:
            augmented_text = self.augmenter._augment_text(sample["text"])
            # Perform a rudimentary check: if augmentation removes all key attributes, revert.
            fe = FeatureEngineer()
            orig_features = fe.extract_features(original_text)
            aug_features = fe.extract_features(augmented_text)
            if not aug_features.get("colors") and orig_features.get("colors"):
                logger.warning("Augmented text lost key color attributes; reverting augmentation.")
                sample["text"] = original_text
            else:
                sample["text"] = augmented_text
            return sample
        except Exception as e:
            logger.warning(f"Runtime augmentation failed: {str(e)}; using original text.")
            sample["text"] = original_text
            return sample
