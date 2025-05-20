import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from typing import Any, Dict
from torchvision.utils import make_grid

class GradientLogger(Callback):
    """Logs gradient statistics during training"""
    def on_after_backward(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        if trainer.global_step % 100 == 0:  # Log every 100 steps
            gradients = []
            for name, param in module.named_parameters():
                if param.grad is not None and "bias" not in name:
                    gradients.append(param.grad.abs().mean())

            if gradients:
                avg_grad = torch.stack(gradients).mean()
                module.log("train/grad_avg", avg_grad, prog_bar=False)

class ExamplePredictionsLogger(Callback):
    """Logs example predictions with input text"""
    def __init__(self, tokenizer, num_examples: int = 4):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.example_batch = None

    def on_validation_start(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        # Store a batch for visualization
        self.example_batch = next(iter(trainer.datamodule.val_dataloader()))

    def on_validation_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        if self.example_batch is None:
            return

        # Get predictions
        with torch.no_grad():
            outputs = module(self.example_batch["input_ids"].to(module.device), 
                           self.example_batch["attention_mask"].to(module.device))

        # Decode examples
        examples = []
        for i in range(min(self.num_examples, len(self.example_batch["text"]))):
            text = self.example_batch["text"][i]
            pred_cats = {k: v[i].argmax().item() for k, v in outputs["category_logits"].items()}
            true_cats = {k: v[i].item() for k, v in self.example_batch["category_labels"].items()}
            pred_sent = {k: torch.sigmoid(v[i]).item() for k, v in outputs["sentiment_logits"].items()}
            
            examples.append({
                "text": text,
                "predictions": {"categories": pred_cats, "sentiment": pred_sent},
                "ground_truth": {"categories": true_cats}
            })

        # Log to logger
        if trainer.logger:
            trainer.logger.log_examples(examples, trainer.global_step)

class CustomProgressBar(pl.callbacks.TQDMProgressBar):
    """Customized progress bar with additional metrics"""
    def get_metrics(self, trainer: pl.Trainer, module: pl.LightningModule) -> Dict[str, Any]:
        items = super().get_metrics(trainer, module)
        items.pop("v_num", None)
        items["lr"] = f"{trainer.optimizers[0].param_groups[0]['lr']:.2e}"
        items["grad_norm"] = f"{module.trainer._accelerator_connector.clip_grad_norm:.2f}"
        return items