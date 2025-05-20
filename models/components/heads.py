import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class MultiTaskHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_categories: int,
        num_sentiments: int,
        category_labels: List[str] = ["color", "material", "style", "fit", "occasion", "weather", "pattern"],
        dropout: float = 0.1
    ):
        super().__init__()
        self.category_labels = category_labels
        self.num_categories = num_categories
        self.num_sentiments = num_sentiments
        
        # Shared feature transformation
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size//2)
        )
        
        # Category-specific heads
        self.category_heads = nn.ModuleDict({
            label: nn.Linear(hidden_size//2, num_classes)
            for label, num_classes in zip(category_labels, num_categories)
        })
        
        # Sentiment heads (one per category)
        self.sentiment_heads = nn.ModuleDict({
            label: nn.Sequential(
                nn.Linear(hidden_size//2, 256),
                nn.GELU(),
                nn.Linear(256, num_sentiments)
            )
            for label in category_labels
        })
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for head in self.category_heads.values():
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)
            
        for head in self.sentiment_heads.values():
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='gelu')
                    nn.init.zeros_(layer.bias)

    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_layer(pooled_output)
        
        category_logits = {
            label: head(features)
            for label, head in self.category_heads.items()
        }
        
        sentiment_logits = {
            label: head(features)
            for label, head in self.sentiment_heads.items()
        }
        
        return category_logits, sentiment_logits

    def get_attention_maps(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """For model interpretability"""
        return {
            label: F.softmax(head(features), dim=-1)
            for label, head in self.category_heads.items()
        }