# src/models/architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from .components.attention import ResidualAttention
from .components.heads import MultiTaskHead

@dataclass
class ModelOutput:
    category_logits: Dict[str, torch.Tensor]
    sentiment_logits: Dict[str, torch.Tensor]
    hidden_states: Optional[Tuple[torch.Tensor]] = None

class ClothingPreferenceModel(nn.Module):
    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        num_categories: int = 7,
        num_sentiments: int = 2,
        hidden_dropout: float = 0.1,
        attention_probs_dropout: float = 0.1,
        use_residual_attention: bool = True
    ):
        super().__init__()
        self.config = {
            "base_model": base_model,
            "num_categories": num_categories,
            "num_sentiments": num_sentiments,
            "hidden_dropout": hidden_dropout,
            "attention_probs_dropout": attention_probs_dropout
        }

        # Base Transformer
        self.bert = DistilBertModel.from_pretrained(
            base_model,
            dropout=hidden_dropout,
            attention_dropout=attention_probs_dropout,
            output_hidden_states=True
        )
        
        # Custom Attention
        self.residual_attention = ResidualAttention(
            hidden_size=self.bert.config.hidden_size,
            num_heads=self.bert.config.num_attention_heads
        ) if use_residual_attention else None

        # Multi-Task Heads
        self.heads = MultiTaskHead(
            hidden_size=self.bert.config.hidden_size,
            num_categories=num_categories,
            num_sentiments=num_sentiments
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Custom weight initialization for enhanced training stability"""
        # Initialize BERT weights using original config
        bert_config = DistilBertConfig.from_pretrained(self.config["base_model"])
        self.bert.init_weights()
        
        # Custom attention initialization
        if self.residual_attention:
            nn.init.xavier_uniform_(self.residual_attention.query.weight)
            nn.init.xavier_uniform_(self.residual_attention.key.weight)
            nn.init.xavier_uniform_(self.residual_attention.value.weight)
        
        # Head initialization
        for head in self.heads.category_heads:
            nn.init.kaiming_normal_(head.weight, mode='fan_out', nonlinearity='relu')
        for head in self.heads.sentiment_heads:
            nn.init.xavier_normal_(head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden_states: bool = False
    ) -> ModelOutput:
        # Base BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Sequence output
        sequence_output = bert_outputs.last_hidden_state
        
        # Apply residual attention
        if self.residual_attention:
            sequence_output = self.residual_attention(
                sequence_output,
                attention_mask
            )

        # Pooling strategy
        pooled_output = self._mean_pooling(sequence_output, attention_mask)
        
        # Get task-specific outputs
        category_logits, sentiment_logits = self.heads(pooled_output)

        return ModelOutput(
            category_logits=category_logits,
            sentiment_logits=sentiment_logits,
            hidden_states=bert_outputs.hidden_states if return_hidden_states else None
        )

    def _mean_pooling(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: Optional[Dict] = None
    ) -> 'ClothingPreferenceModel':
        """Load model from pretrained checkpoint"""
        if config is None:
            config = {}
            
        model = cls(**config)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Handle potential architecture mismatches
        model.load_state_dict(state_dict, strict=False)
        
        # Initialize missing weights
        missing_keys = [k for k in model.state_dict().keys() if k not in state_dict]
        if missing_keys:
            print(f"Initializing missing weights: {missing_keys}")
            model.init_weights()
            
        return model

    def save_pretrained(self, output_dir: str):
        """Save model with proper Transformers format"""
        self.bert.save_pretrained(output_dir)
        torch.save(self.state_dict(), f"{output_dir}/pytorch_model.bin")
        print(f"Model saved to {output_dir}")