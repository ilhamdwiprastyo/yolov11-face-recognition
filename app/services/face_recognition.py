import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms
from app.utils.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)

class FaceRecognizer:
    def __init__(self):
        self.models = model_loader.get_models()
        self.arcface_model = self.models['arcface_model']
        self.class_names = self.models['class_names']
        self.device = self.models['device']
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Face recognizer initialized with {len(self.class_names)} classes")
    
    def preprocess_face(self, face_crop: np.ndarray) -> torch.Tensor:
        """Preprocess face crop for recognition"""
        try:
            # Convert BGR to RGB
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_crop
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Apply transforms
            tensor = self.transform(pil_image)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None
    
    def recognize_face(self, face_crop: np.ndarray, return_embedding: bool = False) -> Dict[str, Any]:
        """Recognize a single face crop"""
        try:
            # Preprocess face
            face_tensor = self.preprocess_face(face_crop)
            if face_tensor is None:
                return None
            
            # Run inference
            with torch.no_grad():
                embeddings, logits = self.arcface_model(face_tensor)
                
                # Get probabilities
                probabilities = F.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get results
                class_id = predicted.item()
                confidence_score = confidence.item()
                person_name = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
                
                result = {
                    'person_name': person_name,
                    'confidence': confidence_score,
                    'class_id': class_id
                }
                
                if return_embedding:
                    result['embedding'] = embeddings.cpu().numpy().flatten().tolist()
                
                return result
                
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None
    
    def recognize_faces(self, face_crops: List[np.ndarray], return_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Recognize multiple face crops"""
        results = []
        
        for i, face_crop in enumerate(face_crops):
            try:
                result = self.recognize_face(face_crop, return_embedding=return_embeddings)
                if result:
                    result['face_id'] = f"face_{i}"
                    results.append(result)
                else:
                    logger.warning(f"Failed to recognize face {i}")
                    
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                
        return results

# Global face recognizer instance
face_recognizer = None

def get_face_recognizer() -> FaceRecognizer:
    """Get global face recognizer instance"""
    global face_recognizer
    if face_recognizer is None:
        face_recognizer = FaceRecognizer()
    return face_recognizer