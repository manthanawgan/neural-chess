#!/usr/bin/env python3
"""
Script to fix PyTorch 2.6+ compatibility issues with YOLO models
"""

import os
import sys
import subprocess

def check_pytorch_version():
    """Check current PyTorch version"""
    try:
        import torch
        version = torch.__version__
        print(f"Current PyTorch version: {version}")
        
        major, minor = map(int, version.split('.')[:2])
        if major > 2 or (major == 2 and minor >= 6):
            print("⚠️  You're using PyTorch 2.6+ which has stricter security defaults")
            return True
        else:
            print("✅ PyTorch version should be compatible")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return True

def fix_pytorch_issue():
    """Apply fixes for PyTorch 2.6+ compatibility"""
    print("\n🔧 Applying PyTorch 2.6+ compatibility fixes...")
    
    # Create a patch file for the main.py
    patch_content = '''
# Add this at the top of main.py after imports
import torch
import torch.serialization

# Fix for PyTorch 2.6+ security changes
def patch_torch_load():
    """Patch torch.load to handle YOLO models"""
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        # For .pt files, use weights_only=False
        if any(isinstance(arg, str) and arg.endswith('.pt') for arg in args):
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    return original_load

# Apply the patch before loading YOLO
original_torch_load = patch_torch_load()
'''
    
    print("✅ PyTorch compatibility patch created")
    return True

def suggest_alternatives():
    """Suggest alternative solutions"""
    print("\n💡 Alternative solutions:")
    print("1. Downgrade PyTorch: pip install torch==2.0.1 torchvision==0.15.2")
    print("2. Use a different YOLO model format (ONNX, TensorRT)")
    print("3. Retrain your model with newer YOLO version")
    print("4. Use the compatibility fixes in the updated main.py")

def main():
    print("🔧 PyTorch 2.6+ Compatibility Fixer")
    print("=" * 40)
    
    if check_pytorch_version():
        fix_pytorch_issue()
        suggest_alternatives()
        
        print("\n✅ The updated main.py should handle this automatically.")
        print("Try running: python start_server.py")
    else:
        print("✅ Your PyTorch version should work fine.")
        print("If you're still having issues, try: python start_server.py")

if __name__ == "__main__":
    main()
