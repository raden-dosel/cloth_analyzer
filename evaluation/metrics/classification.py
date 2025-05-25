import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

class HierarchicalF1(Metric):
    """F1 Score considering parent-child category relationships"""
    def __init__(self, hierarchy: dict, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.hierarchy = hierarchy
        self.add_state("confusion_matrix", default=torch.zeros((len(hierarchy), 2, 2)), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for cat_idx, children in self.hierarchy.items():
            cat_preds = preds[:, cat_idx]
            cat_target = target[:, cat_idx]
            
            # Consider parent-child relationships
            for child in children:
                cat_preds = torch.logical_or(cat_preds, preds[:, child])
                cat_target = torch.logical_or(cat_target, target[:, child])
            
            tp = (cat_preds & cat_target).sum()
            fp = (cat_preds & ~cat_target).sum()
            fn = (~cat_preds & cat_target).sum()
            tn = (~cat_preds & ~cat_target).sum()
            
            self.confusion_matrix[cat_idx] += torch.tensor([[tp, fp], [fn, tn]])

    def compute(self):
        f1_scores = []
        for cat_idx in range(len(self.hierarchy)):
            tp, fp = self.confusion_matrix[cat_idx, 0]
            fn, _ = self.confusion_matrix[cat_idx, 1]
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scores.append(f1)
            
        return torch.mean(torch.stack(f1_scores))

class CategorySpecificity(Metric):
    """Meansure of how specific predictions are within category hierarchy"""
    def __init__(self, hierarchy: dict, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.hierarchy = hierarchy
        self.add_state("total_specificity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for pred, true in zip(preds, target):
            specificity = 0
            for cat_idx, children in self.hierarchy.items():
                if pred[cat_idx] == 1:
                    # Penalize if parent is predicted but no child
                    specificity -= 0.5 * sum(pred[child] == 0 for child in children)
                    
                    # Reward correct child predictions
                    specificity += sum(
                        1.5 * (pred[child] == 1 and true[child] == 1)
                        for child in children
                    )
            self.total_specificity += specificity
            self.total_samples += 1

    def compute(self):
        return self.total_specificity / self.total_samples