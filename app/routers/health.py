from fastapi import APIRouter
from datetime import datetime
import torch
from app.schemas.response import HealthResponse
from app.services.face_detection import get_face_detector
from app.services.face_recognition import get_face_recognizer
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    try:
        # Check models
        models_loaded = {}
        
        try:
            face_detector = get_face_detector()
            models_loaded["face_detector"] = True
        except Exception as e:
            logger.error(f"Face detector not loaded: {e}")
            models_loaded["face_detector"] = False
        
        try:
            face_recognizer = get_face_recognizer()
            models_loaded["face_recognizer"] = True
        except Exception as e:
            logger.error(f"Face recognizer not loaded: {e}")
            models_loaded["face_recognizer"] = False
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        return HealthResponse(
            status="healthy" if all(models_loaded.values()) else "partial",
            timestamp=datetime.now(),
            models_loaded=models_loaded,
            gpu_available=gpu_available,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            models_loaded={},
            gpu_available=False,
            version="1.0.0"
        )