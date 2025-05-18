from torchmetrics import MetricCollection, F1Score, Precision, Recall
import pytorch_lightning as pl


class AttributeMetrics(pl.LightningModule):
    def __init__(self, num_categories):
        super().__init__()
        metrics = MetricCollection({
            "cat_f1": F1Score(num_classes=num_categories, average="macro"),
            "cat_precision": Precision(num_classes=num_categories, average="macro"),
            "cat_recall": Recall(num_classes=num_categories, average="macro")
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        
    def update_metrics(self, preds, labels, mask, stage):
        active_preds = preds[mask == 1]
        active_labels = labels[mask == 1]
        
        if stage == "train":
            self.train_metrics.update(active_preds, active_labels)
        else:
            self.val_metrics.update(active_preds, active_labels)
            
    def log_metrics(self, stage):
        if stage == "train":
            self.log_dict(self.train_metrics.compute(), prog_bar=True)
            self.train_metrics.reset()
        else:
            self.log_dict(self.val_metrics.compute(), prog_bar=True)
            self.val_metrics.reset()