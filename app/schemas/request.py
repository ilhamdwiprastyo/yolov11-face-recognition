from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ProcessingMode(str, Enum):
    DETECT_ONLY = "detect_only"
    RECOGNIZE_ONLY = "recognize_only"
    DETECT_AND_RECOGNIZE = "detect_and_recognize"

class DetectionRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(default=0.5, ge=0.1, le=1.0)
    max_faces: Optional[int] = Field(default=10, ge=1, le=50)
    processing_mode: ProcessingMode = ProcessingMode.DETECT_AND_RECOGNIZE

class VideoProcessingRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(default=0.5, ge=0.1, le=1.0)
    max_faces: Optional[int] = Field(default=10, ge=1, le=50)
    processing_mode: ProcessingMode = ProcessingMode.DETECT_AND_RECOGNIZE
    frame_skip: Optional[int] = Field(default=1, ge=1, le=10)
    max_frames: Optional[int] = Field(default=100, ge=1, le=1000)