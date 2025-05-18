import torch
from transformers import pipeline, __version__ as tf_version

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {tf_version}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Test basic pipeline
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("This dependency setup works!"))