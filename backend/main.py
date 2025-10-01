from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import json
from typing import List, Dict, Any
import os

app = FastAPI(title="Neural Chess API", version="1.0.0")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model with PyTorch 2.6+ compatibility
model_path = "yolov8m.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

def load_yolo_model(model_path):
    """Load YOLO model with compatibility for PyTorch 2.6+ security changes"""
    import torch
    import torch.serialization
    
    try:
        # First try normal loading
        print("🔄 Attempting standard YOLO model loading...")
        return YOLO(model_path)
    except Exception as e:
        print(f"⚠️  Standard loading failed: {e}")
        print("🔄 Attempting to load with weights_only=False...")
        
        # Method 1: Patch torch.load globally
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            model = YOLO(model_path)
            print("✅ Model loaded with weights_only=False")
            return model
        except Exception as e2:
            print(f"⚠️  Patched loading also failed: {e2}")
            print("🔄 Trying alternative approach...")
            
            # Method 2: Use torch.serialization.add_safe_globals
            try:
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                model = YOLO(model_path)
                print("✅ Model loaded with safe globals")
                return model
            except Exception as e3:
                print(f"❌ All loading methods failed: {e3}")
                raise Exception(f"Could not load YOLO model. Please check your model file and PyTorch version. Last error: {e3}")
        finally:
            torch.load = original_load  # Always restore original function

model = load_yolo_model(model_path)
print("YOLO model loaded successfully!")

# Chess piece class mapping (adjust based on your model's classes)
CHESS_PIECE_CLASSES = {
    0: "white_pawn",    # P
    1: "white_rook",    # R
    2: "white_knight",  # N
    3: "white_bishop",  # B
    4: "white_queen",   # Q
    5: "white_king",    # K
    6: "black_pawn",    # p
    7: "black_rook",    # r
    8: "black_knight",  # n
    9: "black_bishop",  # b
    10: "black_queen",  # q
    11: "black_king",   # k
}

def convert_to_chess_notation(piece_class: str) -> str:
    """Convert detected piece class to chess notation"""
    notation_map = {
        "white_pawn": "P", "white_rook": "R", "white_knight": "N", 
        "white_bishop": "B", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_rook": "r", "black_knight": "n", 
        "black_bishop": "b", "black_queen": "q", "black_king": "k"
    }
    return notation_map.get(piece_class, "")

def get_board_position(x_center: float, y_center: float, img_width: int, img_height: int) -> tuple:
    """Convert image coordinates to chess board position (0-7, 0-7)"""
    # Assuming the chess board takes up most of the image
    # This is a simplified approach - you might need to adjust based on your specific use case
    board_size = min(img_width, img_height)
    start_x = (img_width - board_size) // 2
    start_y = (img_height - board_size) // 2
    
    # Calculate which square the piece is in
    square_size = board_size / 8
    col = int((x_center - start_x) / square_size)
    row = int((y_center - start_y) / square_size)
    
    # Clamp to valid board positions
    col = max(0, min(7, col))
    row = max(0, min(7, row))
    
    return (row, col)

@app.get("/")
async def root():
    return {"message": "Neural Chess API is running!"}

@app.post("/analyze-chess-board")
async def analyze_chess_board(file: UploadFile = File(...)):
    """
    Analyze a chess board image and return the detected pieces and their positions
    """
    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        results = model(opencv_image)
        
        # Initialize empty chess board (8x8 grid)
        chess_board = [[None for _ in range(8)] for _ in range(8)]
        detected_pieces = []
        
        # Process detection results
        print(f"Processing {len(results)} detection results...")
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"Found {len(boxes)} detections")
                for box in boxes:
                    # Get bounding box coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    print(f"Detection: class={class_id}, confidence={confidence:.3f}")
                    
                    # Only process high-confidence detections
                    if confidence > 0.5:
                        piece_class = CHESS_PIECE_CLASSES.get(class_id, "unknown")
                        piece_notation = convert_to_chess_notation(piece_class)
                        
                        if piece_notation:
                            # Calculate center of bounding box
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            
                            # Get board position
                            row, col = get_board_position(x_center, y_center, opencv_image.shape[1], opencv_image.shape[0])
                            
                            # Place piece on board
                            chess_board[row][col] = piece_notation
                            
                            detected_pieces.append({
                                "piece": piece_notation,
                                "class": piece_class,
                                "position": {"row": row, "col": col},
                                "confidence": float(confidence),
                                "bbox": {
                                    "x1": float(x1), "y1": float(y1),
                                    "x2": float(x2), "y2": float(y2)
                                }
                            })
        
        print(f"Final result: {len(detected_pieces)} pieces detected")
        print(f"Chess board: {chess_board}")
        
        return {
            "success": True,
            "chess_board": chess_board,
            "detected_pieces": detected_pieces,
            "total_pieces": len(detected_pieces)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded YOLO model"""
    return {
        "model_path": model_path,
        "model_loaded": True,
        "classes": list(CHESS_PIECE_CLASSES.values())
    }

@app.get("/test-chess-board")
async def test_chess_board():
    """Test endpoint that returns a mock chess board to verify frontend integration"""
    # Create a test chess board with some pieces
    test_board = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]
    
    test_pieces = [
        {"piece": "r", "class": "black_rook", "position": {"row": 0, "col": 0}, "confidence": 0.95},
        {"piece": "K", "class": "white_king", "position": {"row": 7, "col": 4}, "confidence": 0.98},
    ]
    
    return {
        "success": True,
        "chess_board": test_board,
        "detected_pieces": test_pieces,
        "total_pieces": len(test_pieces)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
