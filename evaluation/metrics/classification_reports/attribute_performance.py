import torch
import pandas as pd
from torchmetrics import MultilabelPrecision, MultilabelRecall, MultilabelF1Score

def generate_attribute_report(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_names: list,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Generates detailed performance metrics for each clothing attribute
    
    Args:
        logits: Raw model outputs (N samples × M attributes)
        target: Ground truth labels (N samples × M attributes)
        class_names: List of attribute names corresponding to columns
        threshold: Decision boundary for attribute presence
    
    Returns:
        DataFrame with precision, recall, F1, and support per attribute
    """
    # Convert to binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    
    # Initialize metrics
    num_labels = len(class_names)
    metrics = {
        'precision': MultilabelPrecision(num_labels, average='none'),
        'recall': MultilabelRecall(num_labels, average='none'),
        'f1': MultilabelF1Score(num_labels, average='none')
    }
    
    # Calculate metrics
    results = {}
    for metric_name, metric_fn in metrics.items():
        results[metric_name] = metric_fn(preds, target).cpu().numpy()
    
    # Calculate support (true positives + false negatives)
    support = target.sum(dim=0).cpu().numpy()
    
    return pd.DataFrame({
        'attribute': class_names,
        **results,
        'support': support
    }).sort_values('f1', ascending=False)