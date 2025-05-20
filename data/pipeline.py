import os
from typing import Optional
from pathlib import Path
from datasets import load_from_disk, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.processing.tokenization import ClothingTokenizer
from data.validation.schema_validator import DataValidator
from data.generator.generator import SyntheticDataGenerator

class PreferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        max_seq_length: int = 128,
        num_workers: int = 4,
        val_split: float = 0.2,
        synth_ratio: float = 0.3,
        tokenizer: Optional[ClothingTokenizer] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.val_split = val_split
        self.synth_ratio = synth_ratio
        self.tokenizer = tokenizer or ClothingTokenizer()
        self.validator = DataValidator()

        # Path configurations
        self.raw_path = self.data_dir / "raw"
        self.processed_path = self.data_dir / "processed"
        self.synth_path = self.data_dir / "synthetic"

    def prepare_data(self):
        """Run data preparation steps (called once per node)"""
        if not (self.processed_path / "dataset.arrow").exists():
            raw_data = self._load_raw_data()
            synth_data = self._generate_synthetic_data()
            combined = concatenate_datasets([raw_data, synth_data])
            processed = self._preprocess_dataset(combined)
            processed.save_to_disk(self.processed_path)

    def setup(self, stage: Optional[str] = None):
        """Split and prepare datasets (called on each process)"""
        full_dataset = load_from_disk(self.processed_path)
        
        # Train/Val split
        split = full_dataset.train_test_split(test_size=self.val_split)
        self.train_dataset = split["train"]
        self.val_dataset = split["test"]

    def _load_raw_data(self) -> Dataset:
        """Load and validate raw user data"""
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_path}")
            
        datasets = []
        for format in ["csv", "json", "parquet"]:
            for file in self.raw_path.glob(f"**/*.{format}"):
                try:
                    ds = Dataset.from_file(str(file))
                    if self.validator.validate_dataset(ds):
                        datasets.append(ds)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
        
        return concatenate_datasets(datasets) if datasets else Dataset.from_dict({})

    def _generate_synthetic_data(self) -> Dataset:
        """Generate synthetic training data"""
        generator = SyntheticDataGenerator()
        num_samples = int(len(self._load_raw_data()) * self.synth_ratio)
        return generator.generate_dataset(num_samples)

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Process raw data into model-ready format"""
        def process_example(batch):
            tokenized = self.tokenizer.tokenize_with_cleaning(batch["text"])
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["labels"]
            }

        return dataset.map(
            process_example,
            batched=True,
            batch_size=512,
            num_proc=self.num_workers,
            remove_columns=["text"]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # Using val as test for simplicity
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def tokenizer(self) -> ClothingTokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: ClothingTokenizer):
        self._tokenizer = value