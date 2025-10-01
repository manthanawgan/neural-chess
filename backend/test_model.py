#!/usr/bin/env python3
"""
Test script to verify YOLO model loading works correctly
Run this before starting the main server to ensure the model loads properly
"""

import os
import sys

def test_model_loading():
    """Test if the YOLO model can be loaded successfully"""
    model_path = "yolov8m.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("Please ensure your YOLO model file is in the backend directory.")
        return False
    
    try:
        print("🔄 Attempting to load YOLO model...")
        from ultralytics import YOLO
        import torch
        
        # Try standard loading first
        try:
            model = YOLO(model_path)
            print("✅ YOLO model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Standard loading failed: {e}")
            print("🔄 Trying with weights_only=False...")
            
            # Apply the same fix as in main.py
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            try:
                model = YOLO(model_path)
                print("✅ YOLO model loaded with weights_only=False!")
            finally:
                torch.load = original_load
        
        # Test a simple inference to make sure it works
        print("🔄 Testing model inference...")
        import numpy as np
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Dummy image
        results = model(test_image)
        print("✅ Model inference test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure you have the correct YOLO model file")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Try updating PyTorch: pip install torch torchvision --upgrade")
        print("4. If using PyTorch 2.6+, the model might need to be retrained with newer weights")
        return False

if __name__ == "__main__":
    print("🧪 Testing YOLO Model Loading")
    print("=" * 40)
    
    success = test_model_loading()
    
    if success:
        print("\n🎉 Model loading test passed! You can now start the server.")
        sys.exit(0)
    else:
        print("\n💥 Model loading test failed! Please fix the issues above.")
        sys.exit(1)
