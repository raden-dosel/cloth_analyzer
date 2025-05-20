import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

class CategoryF1Score(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Convert to class indices
        preds = preds.argmax(dim=-1)
        
        for c in range(self.num_classes):
            self.true_positives[c] += ((preds == c) & (target == c)).sum()
            self.false_positives[c] += ((preds == c) & (target != c)).sum()
            self.false_negatives[c] += ((preds != c) & (target == c)).sum()

    def compute(self) -> torch.Tensor:
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1.mean()

class SentimentMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.sigmoid()
        self.errors += torch.abs(preds - target).sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.errors / self.total