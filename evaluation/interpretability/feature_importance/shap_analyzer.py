import shap
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import matplotlib.pyplot as plt

class SHAPAnalyzer:
    def __init__(self, model_path, output_dir="interpretability/feature_importance/"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _predict(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=-1).numpy()

    def analyze(self, text, class_idx=1):
        """Generate SHAP values for specified class"""
        explainer = shap.Explainer(self._predict, self.tokenizer)
        shap_values = explainer([text])
        
        # Visualization
        plt.figure()
        shap.plots.text(shap_values[:, :, class_idx], show=False)
        plot_path = self.output_dir / f"shap_{class_idx}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return {
            'values': shap_values.values,
            'plot_path': plot_path,
            'base_value': shap_values.base_values
        }

    def compare_preferences(self, text1, text2):
        """Compare feature importance between two preferences"""
        shap_values1 = self.analyze(text1)
        shap_values2 = self.analyze(text2)
        
        # Generate comparison plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        shap.plots.text(shap_values1['values'], show=False)
        plt.title("First Preference")
        
        plt.subplot(1, 2, 2)
        shap.plots.text(shap_values2['values'], show=False)
        plt.title("Second Preference")
        
        comp_path = self.output_dir / "comparison.png"
        plt.savefig(comp_path)
        plt.close()
        
        return comp_path

class LIMEAnalyzer:
    def __init__(self, model_path):
        from lime.lime_text import LimeTextExplainer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

    def explain(self, text, num_features=5):
        def predictor(texts):
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return torch.softmax(outputs.logits, dim=-1).numpy()
            
        exp = self.explainer.explain_instance(
            text, 
            predictor, 
            num_features=num_features,
            num_samples=100
        )
        
        # Save explanation
        exp_path = Path("interpretability/feature_importance") / f"lime_{text[:10]}.html"
        exp.save_to_file(exp_path)
        
        return {
            'explanation': exp.as_list(),
            'html_path': exp_path
        }

if __name__ == "__main__":
    # SHAP Example
    shap_analyzer = SHAPAnalyzer("your-finetuned-model")
    text = "Flowy summer dress in light colors but not too short"
    shap_results = shap_analyzer.analyze(text)
    
    # LIME Example
    lime_analyzer = LIMEAnalyzer("your-finetuned-model")
    lime_results = lime_analyzer.explain(text)