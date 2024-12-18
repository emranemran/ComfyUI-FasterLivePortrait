import os
import torch
from omegaconf import OmegaConf
from ..FasterLivePortrait.src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

class ModelLoader:
    _instance = None
    _models = {}  # Dictionary to store different model configurations
    
    @classmethod
    def get_model(cls, precision="auto", is_animal=False, paste_back=False):
        # Create a unique key for this configuration
        config_key = f"{precision}_{is_animal}_{paste_back}"
        
        if config_key not in cls._models:
            cls._models[config_key] = cls._load_model(precision, is_animal, paste_back)
        return cls._models[config_key]
    
    @staticmethod
    def _load_model(precision, is_animal, paste_back):
        # Load default config
        config_path = os.path.join(os.path.dirname(__file__), "..", "FasterLivePortrait", "configs", "trt_infer.yaml")
        infer_cfg = OmegaConf.load(config_path)
        
        # Update config based on parameters
        infer_cfg.infer_params.flag_pasteback = paste_back
        
        # Initialize pipeline with TensorRT optimization
        pipeline = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=is_animal)
        
        if precision == "fp16" or (precision == "auto" and torch.cuda.is_available()):
            pipeline.half()  # Convert to FP16 if requested or auto and GPU available
            
        return pipeline
    
    @classmethod
    def unload_model(cls, precision="auto", is_animal=False, paste_back=False):
        config_key = f"{precision}_{is_animal}_{paste_back}"
        if config_key in cls._models:
            del cls._models[config_key]