import torch
from models import AttributeExtractor
from transformers import DistilBertTokenizerFast
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_onnx_model(model_path, onnx_path):
    # Load trained model
    model = AttributeExtractor.load_from_checkpoint(model_path)
    model.eval()

    # Example input
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    sample_input = "Convert this sample text"
    inputs = tokenizer(
        sample_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        export_params=True,
        opset_version=13,
        input_names=["input_ids", "attention_mask"],
        output_names=["categories", "sentiments"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "categories": {0: "batch", 1: "sequence"},
            "sentiments": {0: "batch", 1: "sequence"}
        },
        verbose=False
    )

    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Quantization for production
    quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QUInt8
    )

    return quantized_path

# Usage
export_onnx_model("best_model.ckpt", "clothing_analyzer.onnx")