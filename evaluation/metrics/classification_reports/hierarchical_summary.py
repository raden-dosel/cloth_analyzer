import torch
import pandas as pd
from collections import defaultdict
from torchmetrics import Precision, Recall, F1Score

def hierarchical_summary(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_names: list,
    hierarchy: dict,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Generates aggregated metrics for hierarchical categories
    
    Args:
        logits: Raw model outputs (N samples × M attributes)
        target: Ground truth labels (N samples × M attributes)
        class_names: List of attribute names
        hierarchy: Parent-child relationships {parent: [children]}
        threshold: Decision boundary for attribute presence
    
    Returns:
        DataFrame with aggregated metrics per parent category
    """
    # Convert to binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    
    # Create index mapping
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Build full descendant map
    descendants = defaultdict(list)
    for parent, children in hierarchy.items():
        stack = children.copy()
        while stack:
            child = stack.pop()
            if child in hierarchy:  # Child is itself a parent
                stack.extend(hierarchy[child])
            descendants[parent].append(child)
    
    # Calculate metrics per parent
    results = []
    for parent, children in descendants.items():
        # Get indices of all descendant attributes
        child_indices = [name_to_idx[c] for c in children if c in name_to_idx]
        
        if not child_indices:
            continue
            
        # Aggregate predictions and targets
        parent_preds = (preds[:, child_indices].sum(dim=1) > 0).int()
        parent_targets = (target[:, child_indices].sum(dim=1) > 0).int()
        
        # Calculate metrics
        metrics = {
            'precision': Precision(task='binary'),
            'recall': Recall(task='binary'),
            'f1': F1Score(task='binary')
        }
        
        res = {'parent': parent, 'child_attributes': ', '.join(children)}
        for name, fn in metrics.items():
            res[name] = fn(parent_preds, parent_targets).item()
        
        res['support'] = parent_targets.sum().item()
        results.append(res)
    
    return pd.DataFrame(results).sort_values('f1', ascending=False)