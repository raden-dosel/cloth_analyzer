# classification_reports/category_confusion.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def generate_category_confusion(y_true, y_pred, class_names, example_texts=None):
    """
    y_true: Ground truth labels (n_samples × n_classes)
    y_pred: Predicted labels (n_samples × n_classes)
    class_names: List of attribute names
    example_texts: Optional list of input texts for examples
    """
    errors_mask = np.any(y_true != y_pred, axis=1)
    error_samples = np.where(errors_mask)[0]

    confusion = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(lambda: defaultdict(list))

    for idx in error_samples:
        true_labels = np.where(y_true[idx])[0]
        pred_labels = np.where(y_pred[idx])[0]
        
        # Track false positives
        for pred in pred_labels:
            if pred not in true_labels:
                for true in true_labels:
                    confusion[class_names[true]][class_names[pred]] += 1
                    if example_texts and len(examples[class_names[true]][class_names[pred]]) < 3:
                        examples[class_names[true]][class_names[pred]].append(example_texts[idx])

    # Convert to DataFrame
    rows = []
    for true_cat, preds in confusion.items():
        total = sum(preds.values())
        for pred_cat, count in preds.items():
            example = examples[true_cat][pred_cat][0] if examples else ""
            rows.append({
                "True Category": true_cat,
                "Predicted Category": pred_cat,
                "Count": count,
                "Percentage": count / total * 100,
                "Example Text": example
            })

    return pd.DataFrame(rows).sort_values("Count", ascending=False)