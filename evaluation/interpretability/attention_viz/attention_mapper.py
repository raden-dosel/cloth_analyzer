import os
from pathlib import Path
import torch
from bertviz import head_view, model_view
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class AttentionVisualizer:
    def __init__(self, model_path, output_dir="interpretability/attention_viz/"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _prepare_inputs(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        return inputs

    def generate_head_view(self, text, html_file="attention_head.html"):
        """Interactive multi-head attention visualization"""
        inputs = self._prepare_inputs(text)
        outputs = self.model(**inputs, output_attentions=True)
        
        attention = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        html = head_view(attention, tokens)
        save_path = self.output_dir / html_file
        html.save(str(save_path))
        return save_path

    def generate_model_view(self, text, html_file="attention_flow.html"):
        """Attention pattern flow through layers"""
        inputs = self._prepare_inputs(text)
        outputs = self.model(**inputs, output_attentions=True)
        
        attention = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        html = model_view(attention, tokens)
        save_path = self.output_dir / html_file
        html.save(str(save_path))
        return save_path

    def analyze_negation(self, text):
        """Special analysis for negation patterns"""
        base_text = "I want a formal dress that is not too tight"
        contrast_text = "I want a formal dress that is tight"
        
        paths = {
            'base': self.generate_head_view(base_text, "negation_base.html"),
            'contrast': self.generate_head_view(contrast_text, "negation_contrast.html")
        }
        return paths

if __name__ == "__main__":
    viz = AttentionVisualizer("your-finetuned-model")
    text = "A formal evening dress in dark colors but not too shiny"
    viz.generate_head_view(text)
    viz.generate_model_view(text)