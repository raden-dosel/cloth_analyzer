# src/cli/train.py
import click
import yaml
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pathlib import Path
from typing import Optional
from models.architecture import ClothingPreferenceModel
from training.engine import TrainingModule
from data.pipeline import PreferenceDataModule
from training.utils.loggers import CompositeLogger
from training.utils.callbacks import (
    GradientLogger,
    ExamplePredictionsLogger,
    CustomProgressBar
)

@click.command()
@click.option("--config", default="config/base_config.yaml", help="Path to config file")
@click.option("--resume", help="Path to checkpoint to resume training")
@click.option("--debug", is_flag=True, help="Run in debug mode")
def main(
    config: str = "config/base_config.yaml",
    resume: Optional[str] = None,
    debug: bool = False
):
    """Main training entry point for clothing preference analyzer"""
    
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    model_config_path = Path("config/model_config.yaml")
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    full_config = {**base_config, **model_config}
    
    # Set random seed
    seed_everything(full_config["training"].get("seed", 42), workers=True)

    # Initialize data module
    dm = PreferenceDataModule(
        data_dir=full_config["data"]["data_dir"],
        batch_size=full_config["training"]["batch_size"],
        max_seq_length=full_config["data"]["max_seq_length"],
        num_workers=full_config["data"]["num_workers"],
        val_split=full_config["data"]["validation_split"]
    )
    
    # Initialize model
    if resume:
        # Load from checkpoint
        model = ClothingPreferenceModel.load_from_checkpoint(
            resume,
            config=full_config["model_architecture"]
        )
    else:
        # Create new model
        model = ClothingPreferenceModel(**full_config["model_architecture"])
    
    # Configure logger and callbacks
    logger = CompositeLogger(full_config)
    
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=full_config["training"]["early_stopping_patience"],
            mode="min"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=full_config["logging"]["checkpoint_dir"],
            filename="best-{epoch}-{val_loss:.2f}",
            save_top_k=full_config["logging"]["save_top_k"],
            monitor="val_loss",
            mode="min"
        ),
        GradientLogger(),
        ExamplePredictionsLogger(dm.tokenizer),
        CustomProgressBar()
    ]

    # Configure trainer
    trainer = pl.Trainer(
        accelerator=full_config["hardware"]["accelerator"],
        devices=full_config["hardware"]["devices"],
        precision=full_config["hardware"]["mixed_precision"],
        max_epochs=full_config["training"]["max_epochs"],
        gradient_clip_val=full_config["training"]["gradient_clip_val"],
        accumulate_grad_batches=full_config["training"]["accumulate_grad_batches"],
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=debug,
        deterministic=True,
        benchmark=True
    )

    # Run training
    try:
        trainer.fit(
            model=TrainingModule(model, full_config),
            datamodule=dm,
            ckpt_path=resume
        )
    except Exception as e:
        logger.exception("Training failed!")
        raise

    # Save final model
    if not debug:
        model_dir = Path(full_config["logging"]["model_output_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    main()