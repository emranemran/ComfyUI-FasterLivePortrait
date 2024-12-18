import os
import torch
import tensorrt as trt
from ..FasterLivePortrait.scripts.all_onnx2trt import convert_model

def ensure_tensorrt_model(model_path, onnx_path):
    """
    Ensures TensorRT model exists, converts if necessary
    Args:
        model_path: Path to desired TensorRT model
        onnx_path: Path to source ONNX model
    Returns:
        Path to TensorRT model
    """
    if not os.path.exists(model_path):
        print(f"Converting {onnx_path} to TensorRT...")
        convert_model(onnx_path, model_path)
    return model_path

def convert_all_models():
    """
    Convert all required models to TensorRT format
    Uses the original FasterLivePortrait conversion script
    """
    from ..FasterLivePortrait.scripts.all_onnx2trt import main as convert_all
    convert_all() 