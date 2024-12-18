import os
import sys
import argparse

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_converter import convert_all_models

def main():
    parser = argparse.ArgumentParser(description="Export FasterLivePortrait models to TensorRT")
    parser.add_argument("--force", action="store_true", help="Force reconversion of existing models")
    args = parser.parse_args()
    
    if args.force:
        # Remove existing TRT models if force flag is set
        # Implementation depends on model paths
        pass
        
    # Convert all models to TensorRT format
    convert_all_models()

if __name__ == "__main__":
    main() 