import torch
import numpy as np
from PIL import Image
from .models.model_loader import ModelLoader

class LoadFasterLivePortraitModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "precision": (["auto", "fp32", "fp16"], {"default": "auto"}),
                "is_animal": ("BOOLEAN", {"default": False}),
                "paste_back": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("FASTER_LIVE_PORTRAIT_PIPELINE",)
    FUNCTION = "load_models"
    CATEGORY = "FasterLivePortrait"

    def load_models(self, precision, is_animal, paste_back):
        # Load and return the model pipeline
        return (ModelLoader.get_model(precision, is_animal, paste_back),)

class FasterLivePortraitProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("FASTER_LIVE_PORTRAIT_PIPELINE",),
                "driving_image": ("IMAGE",),
                "source_image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "FasterLivePortrait"

    def process_image(self, pipeline, driving_image, source_image):
        # Convert from ComfyUI image format to PIL
        if isinstance(driving_image, torch.Tensor):
            driving_image = Image.fromarray(np.clip(255. * driving_image.cpu().numpy(), 0, 255).astype(np.uint8))
        if isinstance(source_image, torch.Tensor):
            source_image = Image.fromarray(np.clip(255. * source_image.cpu().numpy(), 0, 255).astype(np.uint8))
            
        # Prepare source image first
        ret = pipeline.prepare_source(source_image, realtime=False)
        if not ret:
            raise ValueError("No face detected in source image!")
            
        # Process the image
        first_frame = True  # Since we're processing one frame at a time
        driving_crop, out_crop, out_org, _ = pipeline.run(
            np.array(driving_image), 
            pipeline.src_imgs[0], 
            pipeline.src_infos[0],
            first_frame=first_frame
        )
        
        if out_crop is None:
            raise ValueError("No face detected in driving image!")
            
        # Return the appropriate output based on paste_back setting
        result = out_org if pipeline.cfg.infer_params.flag_pasteback else out_crop
        
        # Convert back to ComfyUI format
        result_tensor = torch.from_numpy(np.array(result).astype(np.float32) / 255.0)
        return (result_tensor,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LoadFasterLivePortraitModels": LoadFasterLivePortraitModels,
    "FasterLivePortraitProcess": FasterLivePortraitProcess,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFasterLivePortraitModels": "Load FasterLivePortrait Models",
    "FasterLivePortraitProcess": "FasterLivePortrait Process",
} 