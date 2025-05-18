import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from TorchCRF import CRF

class AttributeExtractor(nn.Module):
    def __init__(self, num_categories=7, num_sentiments=2):
        super().__init__()
        self.config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Enhanced classification heads
        self.category_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.config.dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_categories)
        )
        
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.config.dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_sentiments)
        )
        
        # CRF for sequence modeling
        self.crf = CRF(num_categories, batch_first=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Get predictions
        category_logits = self.category_head(sequence_output)
        sentiment_logits = self.sentiment_head(sequence_output)
        
        return category_logits, sentiment_logits