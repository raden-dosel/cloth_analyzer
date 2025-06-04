# data_pipeline.py

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust parent directory in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import pytorch_lightning as pl
from datasets import load_from_disk, Dataset, concatenate_datasets
from torch.utils.data import DataLoader

from data.dataloaders import PreferenceDataset
from data.processing.tokenization import ClothingTokenizer
from data.processing.cleaning import TextCleaner
from data.validation.schema_validator import DataValidator
from data.validation.quality_checker import DataQualityChecker
from data.generator.generator import SyntheticDataGenerator

class PreferenceDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for end-to-end data handling"""
    def __init__(
        self,
        data_dir: Union[str, Path] = "data/",
        batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        synth_ratio: float = 0.3
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.synth_ratio = synth_ratio
        self.cleaner = TextCleaner()
        self.tokenizer = ClothingTokenizer()
        self.validator = DataValidator()
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Setup paths
        self.raw_path = self.data_dir / "raw"
        self.processed_path = self.data_dir / "processed"
        self.synthetic_path = self.data_dir / "synthetic"

    def _validate_and_check_quality(self, dataset: Dataset) -> Dataset:
        """Validate and ensure the quality of the dataset"""
        logger.info("Validating and checking data quality...")
        
        # Convert dataset to a list of dictionaries for validation
        raw_data = dataset.to_dict()
        
        # Step 1: Validate individual samples using DataValidator
        validator = DataValidator(strict=False)
        valid_samples = []
        for sample in raw_data:
            is_valid, validated_sample = validator.validate_sample(sample)
            if is_valid:
                valid_samples.append(validated_sample)
            else:
                logger.warning(f"Invalid sample: {sample['text']} - {validated_sample}")
        
        # Step 2: Run quality checks using DataQualityChecker
        quality_checker = DataQualityChecker()
        quality_report = quality_checker.generate_quality_report(valid_samples)
        logger.info(quality_report)
        
        # Convert valid samples back to a Dataset object
        validated_dataset = Dataset.from_dict(valid_samples)
        return validated_dataset

    def prepare_data(self):
        """Run data preparation steps (called once per node)"""
        logger.info("Starting data preparation...")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.synthetic_path.mkdir(parents=True, exist_ok=True)
        
        dataset_arrow = self.processed_path / "dataset.arrow"
        if not dataset_arrow.exists():
            logger.info("Processed dataset not found - starting processing pipeline.")
            
            # Load raw data
            raw_data = self._load_raw_data()
            if len(raw_data) == 0:
                logger.error("No valid raw data found. Exiting preparation.")
                return
            
            # Validate and check quality of raw data
            validated_data = self._validate_and_check_quality(raw_data)
            
            # Generate synthetic data
            synth_data = self._generate_synthetic_data(len(validated_data))
            
            # Combine datasets
            combined = concatenate_datasets([validated_data, synth_data])
            
            # Preprocess and save to disk
            processed = self._preprocess_dataset(combined)
            processed.save_to_disk(self.processed_path)
            logger.info("Data processing complete and saved to disk.")

    def setup(self, stage: Optional[str] = None):
        """Split data into train/val/test (called on each GPU)"""
        logger.info("Setting up datasets for training, validation, and testing...")
        full_dataset = load_from_disk(self.processed_path)
        
        # Perform stratified split based on main_category
        train_val = full_dataset.train_test_split(
            test_size=self.val_split + self.test_split,
            stratify_by_column="main_category"
        )
        
        val_test = train_val["test"].train_test_split(
            test_size=self.test_split/(self.val_split + self.test_split),
            stratify_by_column="main_category"
        )
        
        self.train_dataset = PreferenceDataset(
            train_val["train"],
            self.tokenizer,
            self.max_length,
            augment=True
        )
        self.val_dataset = PreferenceDataset(
            val_test["train"],
            self.tokenizer,
            self.max_length
        )
        self.test_dataset = PreferenceDataset(
            val_test["test"],
            self.tokenizer,
            self.max_length
        )

    def _load_raw_data(self) -> Dataset:
        """Load and validate raw user data from directory"""
        datasets = []
        # Supported formats
        formats = {
            "csv": lambda p: Dataset.from_csv(str(p)),
            "json": lambda p: Dataset.from_json(str(p)),
            "parquet": lambda p: Dataset.from_parquet(str(p))
        }
        
        for fmt, loader in formats.items():
            for file in self.raw_path.glob(f"**/*.{fmt}"):
                try:
                    ds = loader(file)
                    if self.validator.validate_dataset(ds):
                        # Add main category field for stratified splitting
                        ds = ds.map(self._add_main_category)
                        datasets.append(ds)
                        logger.info(f"Loaded and validated file: {file}")
                    else:
                        logger.warning(f"Dataset from {file} failed validation.")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
        
        if datasets:
            combined_ds = concatenate_datasets(datasets)
            logger.info(f"Combined raw dataset size: {len(combined_ds)}")
            return combined_ds
        else:
            logger.error("No datasets loaded from raw data!")
            return Dataset.from_dict({})

    def _add_main_category(self, sample: Dict) -> Dict:
        """Determine main category for stratified splitting"""
        if sample["attributes"].get("occasion"):
            sample["main_category"] = "occasion"
        elif sample["attributes"].get("style"):
            sample["main_category"] = "style"
        else:
            sample["main_category"] = "other"
        return sample

    def _generate_synthetic_data(self, num_real: int) -> Dataset:
        """Generate synthetic training data"""
        num_synth = max(1000, int(num_real * self.synth_ratio))
        synth_file = self.synthetic_path / "synthetic_data.arrow"
        if synth_file.exists():
            logger.info("Loading existing synthetic data.")
            return load_from_disk(synth_file)
        
        logger.info("Generating new synthetic data...")
        generator = SyntheticDataGenerator()
        synthetic = generator.generate_dataset(num_synth)
        synthetic.save_to_disk(synth_file)
        logger.info(f"Synthetic data generated with {num_synth} samples.")
        return synthetic

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Process raw data into model-ready format using cleaning and tokenization"""
        logger.info("Preprocessing dataset: cleaning and tokenizing text...")
        
        def preprocess_function(batch):
            # Clean the text using the cleaner
            cleaned_text = self.cleaner.clean(batch["text"])
            # Tokenize the cleaned text
            return self.tokenizer.tokenize_with_cleaning(
                cleaned_text,
                max_length=self.max_length
            )
        
        processed_ds = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=256,
            num_proc=self.num_workers,
            remove_columns=["text"]
        )
        logger.info("Preprocessing complete.")
        return processed_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._dynamic_padding
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._dynamic_padding
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._dynamic_padding
        )

    def _dynamic_padding(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for dynamic padding"""
        inputs = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} 
                  for item in batch]
        labels = [item["labels"] for item in batch if "labels" in item] or None
        
        padded = self.tokenizer.pad(
            inputs,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )
        if labels:
            padded["labels"] = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100  # Typically ignore index for loss computation
            )
        return padded

    @property
    def num_categories(self) -> int:
        """Number of clothing categories in the dataset"""
        if not hasattr(self, "_num_categories"):
            self._num_categories = len(
                self.train_dataset.dataset.features["attributes"]["category"].names
            )
        return self._num_categories
