import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Optional, Dict, Any
import torch
import wandb

class CustomWandbLogger(WandbLogger):
    """Extended WandB logger with example logging"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.example_table = None

    def log_examples(self, examples: Dict, step: int) -> None:
        """Log text examples with predictions"""
        table = wandb.Table(columns=["Text", "Predicted", "Ground Truth"])
        
        for ex in examples:
            table.add_data(
                ex["text"],
                str(ex["predictions"]),
                str(ex["ground_truth"])
            )
            
        self.experiment.log({
            "examples": table,
            "global_step": step
        })

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Handle custom metric types"""
        filtered_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().item()
            filtered_metrics[k] = v
            
        super().log_metrics(filtered_metrics, step)

class CompositeLogger(pl.loggers.LoggerCollection):
    """Logs to multiple services simultaneously"""
    def __init__(self, config: Dict):
        loggers = []
        
        if config["logging"]["experiment_tracker"] == "wandb":
            loggers.append(CustomWandbLogger(
                project=config["logging"]["wandb"]["project"],
                entity=config["logging"]["wandb"]["entity"],
                log_model="all"
            ))
            
        loggers.append(TensorBoardLogger(
            save_dir=config["logging"]["log_dir"],
            name="tensorboard"
        ))
        
        super().__init__(loggers)

    def log_hyperparams(self, params: Dict) -> None:
        for logger in self._logger_iterable:
            logger.log_hyperparams(params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for logger in self._logger_iterable:
            logger.log_metrics(metrics, step)

    def log_examples(self, examples: Dict, step: int) -> None:
        for logger in self._logger_iterable:
            if isinstance(logger, CustomWandbLogger):
                logger.log_examples(examples, step)