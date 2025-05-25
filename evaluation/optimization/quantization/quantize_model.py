import torch
from torch.quantization import quantize_dynamic, prepare_qat, convert
from torch.quantization.qconfig import get_default_qat_qconfig
from pathlib import Path

class ModelQuantizer:
    def __init__(self, model, config_path="configs/optimization.yaml"):
        self.model = model
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_config(self, path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)['quantization']
    
    def dynamic_quantization(self, output_dir="quantization/"):
        """Post-training dynamic quantization"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Quantize model
        quantized_model = quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
        
        # Save artifacts
        torch.save(quantized_model.state_dict(), f"{output_dir}/quantized_weights.pth")
        print(f"Quantized model saved to {output_dir}")
        return quantized_model

    def qat(self, train_loader, epochs=3):
        """Quantization-Aware Training"""
        self.model.train()
        self.model.qconfig = get_default_qat_qconfig('fbgemm')
        prepared_model = prepare_qat(self.model)
        
        # Fine-tuning loop
        optimizer = torch.optim.AdamW(prepared_model.parameters(), lr=1e-5)
        for epoch in range(epochs):
            for batch in train_loader:
                outputs = prepared_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        quantized_model = convert(prepared_model)
        return quantized_model

if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("your-finetuned-model")
    quantizer = ModelQuantizer(model)
    quantizer.dynamic_quantization()