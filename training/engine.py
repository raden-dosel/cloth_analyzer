import pytorch_lightning as pl
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict, List, Optional
from .loss import DynamicWeightedLoss
from .metrics import CategoryF1Score, SentimentMAE

class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        class_weights: Optional[Dict[str, List[float]]] = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.class_weights = class_weights or {}
        
        # Initialize components
        self.loss_fn = DynamicWeightedLoss(
            num_categories=len(config["model"]["category_labels"]),
            class_weights=self._prepare_class_weights(),
            label_smoothing=config["training"].get("label_smoothing", 0.1)
        )
        
        self.metrics = {
            "category_f1": CategoryF1Score(num_classes=config["model"]["num_categories"]),
            "sentiment_mae": SentimentMAE()
        }
        
        # Save hyperparameters for Lightning
        self.save_hyperparameters(ignore=["model"])

    def _prepare_class_weights(self) -> torch.Tensor:
        """Convert class weights to tensor"""
        weights = []
        for label in self.config["model"]["category_labels"]:
            if label in self.class_weights:
                weights.append(torch.tensor(self.class_weights[label]))
            else:
                weights.append(torch.ones(self.config["model"]["num_categories"]))
        return torch.stack(weights).to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        return self.model(input_ids, attention_mask)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(outputs, batch)
        self._log_metrics(outputs, batch, "train")
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(outputs, batch)
        self._log_metrics(outputs, batch, "val")
        return {"val_loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"]["weight_decay"],
            eps=1e-8
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config["training"]["max_epochs"],
            eta_min=self.config["training"].get("min_lr", 1e-6)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            },
            "monitor": "val_loss"
        }

    def _log_metrics(
        self,
        outputs: Dict,
        batch: Dict,
        prefix: str
    ) -> None:
        """Log metrics for both training and validation"""
        # Category predictions
        category_preds = torch.stack(
            [outputs["category_logits"][cat] for cat in self.config["model"]["category_labels"]],
            dim=1
        ).argmax(-1)
        
        # Sentiment predictions
        sentiment_preds = torch.stack(
            [outputs["sentiment_logits"][cat] for cat in self.config["model"]["category_labels"]],
            dim=1
        ).sigmoid()
        
        # Update and log metrics
        for name, metric in self.metrics.items():
            if "category" in name:
                metric.update(category_preds, batch["category_labels"])
            else:
                metric.update(sentiment_preds, batch["sentiment_labels"])
            
            self.log(
                f"{prefix}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True
            )

        # Log learning rate
        if prefix == "train":
            self.log(
                "lr", 
                self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )