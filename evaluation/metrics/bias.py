from torchmetrics import Metric
import torch

class DemographicParityDifference(Metric):
    """Fairness metric across demographic groups"""
    def __init__(self, demographic_categories: list, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.demographic_categories = demographic_categories
        self.add_state("group_counts", default=torch.zeros((len(demographic_categories), 2)), 
                      dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, sensitive_attributes: torch.Tensor):
        for i, category in enumerate(self.demographic_categories):
            mask = sensitive_attributes == category
            pos_preds = preds[mask].sum()
            total = mask.sum()
            self.group_counts[i] += torch.tensor([pos_preds, total])

    def compute(self):
        parity_diffs = []
        global_rate = self.group_counts[:, 0].sum() / self.group_counts[:, 1].sum()
        
        for i in range(len(self.demographic_categories)):
            group_rate = self.group_counts[i, 0] / self.group_counts[i, 1]
            parity_diffs.append(torch.abs(group_rate - global_rate))
            
        return torch.mean(torch.stack(parity_diffs))