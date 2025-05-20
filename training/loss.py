import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class DynamicWeightedLoss(nn.Module):
    def __init__(
        self,
        num_categories: int,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.num_categories = num_categories
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        
        # Initialize weights
        self.class_weights = class_weights if class_weights is not None \
            else torch.ones(num_categories)
            
        # Base loss functions
        self.category_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )
        self.sentiment_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        # Category loss
        category_logits = torch.stack(
            [outputs["category_logits"][cat] for cat in batch["category_labels"]],
            dim=1
        )
        category_loss = self._focal_loss(
            category_logits,
            batch["category_labels"],
            self.category_loss
        )
        
        # Sentiment loss
        sentiment_logits = torch.stack(
            [outputs["sentiment_logits"][cat] for cat in batch["sentiment_labels"]],
            dim=1
        )
        sentiment_loss = self.sentiment_loss(
            sentiment_logits,
            batch["sentiment_labels"].float()
        )
        
        # Apply sample weights
        sentiment_loss = sentiment_loss * batch.get("sample_weights", 1.0)
        
        # Combine losses
        total_loss = (
            self.config["training"]["category_weight"] * category_loss +
            self.config["training"]["sentiment_weight"] * sentiment_loss.mean()
        )
        
        return total_loss

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        base_loss: nn.Module
    ) -> torch.Tensor:
        """Add focal weighting to base loss"""
        ce_loss = base_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.focal_gamma * ce_loss).mean()