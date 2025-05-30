import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from config.settings import settings
from app.schemas.response import ProcessedFace
from app.utils.logger import get_logger

logger = get_logger(__name__)

def validate_image(filename: str) -> bool:
    """Validate image file extension"""
    if not filename:
        return False
    
    extension = Path(filename).suffix.lower().lstrip('.')
    return extension in settings.allowed_image_extensions_list  # Use property

def validate_video(filename: str) -> bool:
    """Validate video file extension"""
    if not filename:
        return False
    
    extension = Path(filename).suffix.lower().lstrip('.')
    return extension in settings.allowed_video_extensions_list  # Use property

def save_processed_image(image: np.ndarray, faces: List[ProcessedFace], image_id: str) -> str:
    """Save image with face detection and recognition annotations"""
    
    try:
        annotated_image = image.copy()
        
        for face in faces:
            # Get bounding box
            bbox = face.detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_image,
                         (bbox.x1, bbox.y1),
                         (bbox.x2, bbox.y2),
                         (0, 255, 0), 2)
            
            # Prepare text
            if face.recognition and face.recognition.confidence > 0.5:
                text = f"{face.recognition.person_name}: {face.recognition.confidence:.2f}"
                color = (0, 255, 0)  # Green for recognized
            else:
                text = f"Unknown: {face.detection.confidence:.2f}"
                color = (0, 0, 255)  # Red for unknown
            
            # Draw text background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(annotated_image,
                         (bbox.x1, bbox.y1 - text_height - 10),
                         (bbox.x1 + text_width, bbox.y1),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_image, text,
                       (bbox.x1, bbox.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence on bottom right
            conf_text = f"Det: {face.detection.confidence:.2f}"
            cv2.putText(annotated_image, conf_text,
                       (bbox.x2 - 80, bbox.y2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save annotated image
        output_path = Path(settings.output_dir) / f"{image_id}_processed.jpg"
        cv2.imwrite(str(output_path), annotated_image)
        
        logger.info(f"Saved processed image: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error saving processed image: {e}")
        return None

def resize_image(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized