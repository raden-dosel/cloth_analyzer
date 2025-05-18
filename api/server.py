from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast
from typing import List, Dict
import torch

app = FastAPI()

# Configuration
MODEL_PATH = "clothing_analyzer_quantized.onnx"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
ort_session = ort.InferenceSession(MODEL_PATH)

class AnalysisRequest(BaseModel):
    text: str
    threshold: float = 0.7

class AttributePrediction(BaseModel):
    phrase: str
    category: str
    sentiment: str
    confidence: float

@app.post("/analyze", response_model=List[AttributePrediction])
async def analyze_text(request: AnalysisRequest):
    try:
        # Preprocess input
        inputs = tokenizer(
            request.text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # ONNX Inference
        outputs = ort_session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        )
        
        # Postprocessing
        category_probs = torch.softmax(torch.tensor(outputs[0]), dim=-1)
        sentiment_probs = torch.softmax(torch.tensor(outputs[1]), dim=-1)
        
        # Convert to JSON response
        return process_onnx_outputs(
            inputs["input_ids"][0],
            category_probs[0],
            sentiment_probs[0],
            request.threshold
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_onnx_outputs(input_ids, category_probs, sentiment_probs, threshold):
    # Implementation similar to previous process_outputs
    # with confidence threshold filtering
    processed_outputs = []
    for i in range(len(input_ids)):
        if category_probs[i] > threshold:
            processed_outputs.append({
                "phrase": tokenizer.decode(input_ids[i]),
                "category": "clothing",
                "sentiment": "positive" if sentiment_probs[i] > 0.5 else "negative",
                "confidence": float(category_probs[i])
            })
    return processed_outputs