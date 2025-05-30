import cv2
import numpy as np
import urllib.request
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class YOLOFaceDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        """Initialize YOLO face detector"""
        self.model_path = model_path or settings.yolo_model_path
        self.conf_threshold = conf_threshold
        
        # Check if model exists, if not then download it
        if not Path(self.model_path).exists():
            logger.warning(f"YOLOv11 model not found at: {self.model_path}")
            self.download_model(self.model_path)
        
        # Load model
        self.model = YOLO(self.model_path)
        logger.info(f"YOLOv11 face detector loaded from: {self.model_path}")
    
    def download_model(self, path: str):
        """Download the YOLOv11 face detection model"""
        url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt"
        
        # Create directory if it doesn't exist
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading YOLOv11 face model from: {url}")
        
        try:
            urllib.request.urlretrieve(url, path)
            logger.info("Model successfully downloaded")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise FileNotFoundError(f"Failed to download model from {url}")
    
    def detect_faces(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """Detect faces in image and return face crops with bounding boxes"""
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
            
        if image is None:
            return []
        
        # Run detection
        results = self.model(image, conf=conf_threshold, verbose=False)
        faces = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Ensure bbox is within image bounds
                    h, w = image.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    
                    # Extract face crop
                    face_crop = image[y1:y2, x1:x2]
                    
                    # Check minimum size
                    min_size = settings.min_face_size
                    if face_crop.shape[0] >= min_size and face_crop.shape[1] >= min_size:
                        faces.append({
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                            'confidence': confidence,
                            'face_crop': face_crop
                        })
        
        # Sort by confidence and limit number of faces
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        return faces[:settings.max_faces_per_image]

# Global face detector instance
face_detector = None

def get_face_detector() -> YOLOFaceDetector:
    """Get global face detector instance"""
    global face_detector
    if face_detector is None:
        face_detector = YOLOFaceDetector()
    return face_detector