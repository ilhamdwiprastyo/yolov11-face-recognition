from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List, Union
import torch
import os

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    debug: bool = True
    
    # Model Paths
    yolo_model_path: str = "./models/yolov11s-face.pt"
    arcface_model_path: str = "./models/arcface_resnet50_final_model.pth"
    classes_file_path: str = "./models/classes.txt"
    
    # Detection Settings
    detection_confidence: float = 0.5
    min_face_size: int = 50
    max_faces_per_image: int = 10
    
    # Upload Settings - Fixed parsing
    max_file_size: str = "50MB"
    allowed_image_extensions: str = "jpg,jpeg,png,bmp"  # Changed to string
    allowed_video_extensions: str = "mp4,avi,mov,mkv"  # Changed to string
    
    # Security
    secret_key: str = "your-secret-key-here"
    access_token_expire_minutes: int = 30
    
    # Directories
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    
    # Device Configuration
    device: str = "auto"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories
        for dir_path in [self.upload_dir, self.output_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Auto-detect optimal device
        self.device = self._get_optimal_device()
    
    def _get_optimal_device(self) -> str:
        """Auto-detect optimal device"""
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return "mps"  # Mac M1
            elif torch.cuda.is_available():
                return "cuda"  # NVIDIA GPU
            else:
                return "cpu"   # CPU fallback
        return self.device
    
    @property
    def allowed_image_extensions_list(self) -> List[str]:
        """Convert comma-separated string to list"""
        return [ext.strip() for ext in self.allowed_image_extensions.split(",")]
    
    @property
    def allowed_video_extensions_list(self) -> List[str]:
        """Convert comma-separated string to list"""
        return [ext.strip() for ext in self.allowed_video_extensions.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert file size string to bytes"""
        size_str = self.max_file_size.upper()
        if size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        elif size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        else:
            return int(size_str)

settings = Settings()