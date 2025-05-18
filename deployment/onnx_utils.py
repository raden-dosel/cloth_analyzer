import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.optimizer import optimize_model
import numpy as np

def quantize_onnx_model(input_path: str, output_path: str):
    """Quantize ONNX model to INT8"""
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8,
        optimize_model=True,
        use_external_data_format=False
    )

def validate_onnx_model(model_path: str):
    """Check ONNX model validity"""
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(f"✅ Model {model_path} is valid")
    print(f"Inputs: {[i.name for i in model.graph.input]}")
    print(f"Outputs: {[o.name for o in model.graph.output]}")

def create_onnx_inputs(tokenizer, text: str):
    """Create ONNX-compatible input tensors"""
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=128
    )
    return {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }