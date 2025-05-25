from torchmetrics import Metric
import torch

class ContextualizedMAE(Metric):
    """MAE weighted by category importance"""
    def __init__(self, category_weights: dict, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.category_weights = category_weights
        self.add_state("weighted_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        errors = torch.abs(preds - target)
        weights = torch.tensor([
            self.category_weights[cat]
            for cat in sorted(self.category_weights)
        ]).to(errors.device)
        
        self.weighted_errors += torch.sum(errors * weights)
        self.total_weight += torch.sum(weights)

    def compute(self):
        return self.weighted_errors / self.total_weight

class SentimentConsistency(Metric):
    """Consistency between category predictions and sentiment scores"""
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("consistent", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, category_preds: torch.Tensor, sentiment_scores: torch.Tensor):
        # Category predicted but sentiment is neutral/opposite
        inconsistent = (
            (category_preds == 1) & 
            (torch.abs(sentiment_scores) < 0.3)
        ).sum()
        
        self.consistent += (sentiment_scores.numel() - inconsistent)
        self.total += sentiment_scores.numel()

    def compute(self):
        return self.consistent.float() / self.total