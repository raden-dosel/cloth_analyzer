import torch
import torch.nn.utils.prune as prune
from pathlib import Path

class ModelPruner:
    def __init__(self, model, config_path="configs/optimization.yaml"):
        self.model = model
        self.config = self._load_config(config_path)['pruning']
        self._prepare_model()

    def _load_config(self, path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    def _prepare_model(self):
        """Identify prunable parameters"""
        self.parameters_to_prune = [
            (module, "weight") 
            for module in self.model.modules() 
            if isinstance(module, torch.nn.Linear)
        ]

    def iterative_pruning(self, output_dir="pruning/"):
        """Iterative magnitude pruning"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for step in range(self.config['steps']):
            # Prune globally across all layers
            prune.global_unstructured(
                self.parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config['amount_per_step'],
            )
            
            # Remove pruning reparameterization
            for module, _ in self.parameters_to_prune:
                prune.remove(module, 'weight')
            
            # Save checkpoint
            sparsity = self._calculate_sparsity()
            torch.save(
                self.model.state_dict(),
                f"{output_dir}/step_{step}_sparsity_{sparsity:.2f}.pth"
            )
            print(f"Step {step}: Sparsity {sparsity:.2%}")

    def _calculate_sparsity(self):
        total_zeros = 0
        total_elements = 0
        for module, _ in self.parameters_to_prune:
            total_zeros += torch.sum(module.weight == 0)
            total_elements += module.weight.nelement()
        return total_zeros / total_elements

if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("your-finetuned-model")
    pruner = ModelPruner(model)
    pruner.iterative_pruning()