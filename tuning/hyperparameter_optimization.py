import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent_folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.training.train import TrainingModule
import joblib
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_from_disk

def objective(trial):
    # Suggest hyperparameters
    config = {
        "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "gradient_clip": trial.suggest_float("gradient_clip", 0.1, 1.0)
    }

    # Load data with dynamic batch size
    dataset = load_from_disk("data/synthetic_dataset")
    train_loader = DataLoader(dataset.train, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset.val, batch_size=config["batch_size"]*2)

    # Initialize model with suggested params
    model = TrainingModule(
        dropout=config["dropout"],
        class_weights=dataset.class_weights
    )

    # Configure trainer with pruning
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        gradient_clip_val=config["gradient_clip"],
        enable_progress_bar=False
    )

    # Execute training
    trainer.fit(model, train_loader, val_loader)
    
    return trainer.callback_metrics["val_loss"].item()

# Optimization setup
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
)

study.optimize(objective, n_trials=50, timeout=3600)

# Output best parameters
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save optimization results
joblib.dump(study, "optimization_study.pkl")