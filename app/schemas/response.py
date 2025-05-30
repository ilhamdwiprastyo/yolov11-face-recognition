from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class FaceDetection(BaseModel):
    bbox: BoundingBox
    confidence: float
    face_id: Optional[str] = None

class FaceRecognition(BaseModel):
    person_name: str
    confidence: float
    class_id: int
    embedding: Optional[List[float]] = None

class ProcessedFace(BaseModel):
    detection: FaceDetection
    recognition: Optional[FaceRecognition] = None

class ImageProcessingResult(BaseModel):
    image_id: str
    filename: str
    processing_time: float
    faces_detected: int
    faces_recognized: int
    faces: List[ProcessedFace]
    output_image_path: Optional[str] = None
    timestamp: datetime

class VideoProcessingResult(BaseModel):
    video_id: str
    filename: str
    total_frames: int
    processed_frames: int
    processing_time: float
    faces_detected: int
    faces_recognized: int
    output_video_path: Optional[str] = None
    frame_results: List[Dict[str, Any]]
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    gpu_available: bool
    version: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime