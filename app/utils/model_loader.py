import torch
import torch.nn as nn
from torchvision.models import resnet50
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from config.settings import settings
from app.utils.logger import get_logger
import platform
import os

logger = get_logger(__name__)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(3.14159265359 - m))
        self.mm = torch.sin(torch.tensor(3.14159265359 - m)) * m

    def forward(self, input, label=None):
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input), 
                                           torch.nn.functional.normalize(self.weight))
        if label is None:
            return cosine * self.s
            
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, dropout=0.5):
        super(ArcFaceModel, self).__init__()
        # Use ResNet50 as backbone
        self.backbone = resnet50(pretrained=True)
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # ArcFace classifier
        self.arcface = ArcMarginProduct(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        
    def forward(self, x, labels=None):
        # Extract features
        features = self.backbone(x)
        # Get embeddings
        embeddings = self.embedding(features)
        # ArcFace classification
        logits = self.arcface(embeddings, labels)
        return embeddings, logits

class ModelLoader:
    def __init__(self):
        self.device = self._get_device()
        self.arcface_model: Optional[ArcFaceModel] = None
        self.class_names: List[str] = []
        self.num_classes: int = 0
        
        logger.info(f"Mac M1 Detection: {self._is_mac_m1()}")
        logger.info(f"Selected device: {self.device}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        
    def _is_mac_m1(self) -> bool:
        """Check if running on Mac M1"""
        return platform.system() == "Darwin" and platform.processor() == "arm"
    
    def _get_device(self) -> torch.device:
        """Get optimal device for Mac M1"""
        if torch.backends.mps.is_available():
            # Mac M1 dengan MPS support
            logger.info("Using Metal Performance Shaders (MPS) for acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            # NVIDIA GPU (tidak ada di Mac M1)
            logger.info("Using CUDA GPU")
            return torch.device("cuda")
        else:
            # CPU fallback
            logger.info("Using CPU (consider upgrading PyTorch for MPS support)")
            return torch.device("cpu")
        
    def load_class_names(self) -> List[str]:
        """Load class names from file"""
        try:
            classes_path = Path(settings.classes_file_path)
            if not classes_path.exists():
                raise FileNotFoundError(f"Classes file not found: {classes_path}")
                
            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
                
            logger.info(f"Loaded {len(class_names)} class names")
            return class_names
            
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            raise
    
    def load_arcface_model(self) -> ArcFaceModel:
        """Load ArcFace model for face recognition"""
        try:
            # Load class names first
            self.class_names = self.load_class_names()
            self.num_classes = len(self.class_names)
            
            # Create model
            model = ArcFaceModel(self.num_classes, embedding_dim=512)
            
            # Load weights
            model_path = Path(settings.arcface_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"ArcFace model not found: {model_path}")
                
            # Load checkpoint with Mac M1 compatibility
            if self.device.type == "mps":
                # For MPS, load to CPU first then move to MPS
                checkpoint = torch.load(model_path, map_location="cpu")
            else:
                checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            # Move to device after loading
            model.to(self.device)
            model.eval()
            
            # Mac M1 specific optimizations
            if self.device.type == "mps":
                # Enable MPS optimizations
                model = torch.jit.script(model) if hasattr(torch.jit, 'script') else model
                logger.info("Applied MPS optimizations")
            
            logger.info(f"ArcFace model loaded successfully")
            logger.info(f"Device: {self.device}")
            logger.info(f"Classes: {self.num_classes}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading ArcFace model: {e}")
            # Fallback to CPU if MPS fails
            if self.device.type == "mps":
                logger.warning("MPS failed, falling back to CPU")
                self.device = torch.device("cpu")
                return self.load_arcface_model()
            raise
    
    def get_models(self) -> Dict[str, Any]:
        """Get all loaded models"""
        if self.arcface_model is None:
            self.arcface_model = self.load_arcface_model()
            
        return {
            'arcface_model': self.arcface_model,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'device': self.device
        }

# Global model loader instance
model_loader = ModelLoader()