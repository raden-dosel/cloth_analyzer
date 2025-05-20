import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ResidualAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, (
            "Hidden size must be divisible by num_heads"
        )
        
        # Projection layers
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            residual = hidden_states
            
            # Project to query/key/value
            query = self._shape(self.query(hidden_states))
            key = self._shape(self.key(hidden_states))
            value = self._shape(self.value(hidden_states))
            
            # Attention scores
            scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
            
            # Apply attention mask
            if attention_mask is not None:
                scores = scores + attention_mask.unsqueeze(1).unsqueeze(2)
                
            # Attention probabilities
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            
            # Contextualized embeddings
            context = torch.matmul(probs, value)
            context = self._unshape(context)
            
            # Final projection and residual
            output = self.out_proj(context)
            output = self.layer_norm(output + residual)
            
            return output

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

    def _unshape(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.hidden_size)