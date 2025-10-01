#!/usr/bin/env python3
"""
Enhanced server startup script with model testing
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'ultralytics', 'fastapi', 'uvicorn', 'torch', 'torchvision', 
        'opencv-python', 'pillow', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found!")
    return True

def test_model():
    """Test if the YOLO model can be loaded"""
    print("\n🧪 Testing YOLO model...")
    
    if not os.path.exists("yolov8m.pt"):
        print("❌ Model file 'yolov8m.pt' not found!")
        print("Please ensure your YOLO model file is in the backend directory.")
        return False
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, "test_model.py"], 
                               capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Model test passed!")
            return True
        else:
            print(f"❌ Model test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model test timed out")
        return False
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Neural Chess Backend Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    print("🎯 Neural Chess Backend Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💥 Please install missing dependencies first!")
        sys.exit(1)
    
    # Test model loading
    if not test_model():
        print("\n💥 Model loading failed! Please check your YOLO model file.")
        print("You can try:")
        print("1. Ensure yolov8m.pt is in the backend directory")
        print("2. Update PyTorch: pip install torch torchvision --upgrade")
        print("3. Re-download or retrain your YOLO model")
        sys.exit(1)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()