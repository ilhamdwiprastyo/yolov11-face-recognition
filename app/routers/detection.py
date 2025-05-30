from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import cv2
import numpy as np
from pathlib import Path
import uuid
import time
import aiofiles
from typing import List, Optional
from datetime import datetime

from app.schemas.request import DetectionRequest, VideoProcessingRequest, ProcessingMode
from app.schemas.response import (
    ImageProcessingResult, VideoProcessingResult, ProcessedFace, 
    FaceDetection, FaceRecognition, BoundingBox, ErrorResponse
)
from app.services.face_detection import get_face_detector
from app.services.face_recognition import get_face_recognizer
from app.services.video_processor import get_video_processor
from app.utils.image_utils import validate_image, validate_video, save_processed_image
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["detection"])

# Dependency untuk validasi file size
async def validate_file_size(file: UploadFile):
    if file.size and file.size > settings.max_file_size_bytes:  # Use property
        raise HTTPException(status_code=413, detail="File too large")
    return file

@router.post("/detect/image", response_model=ImageProcessingResult)
async def detect_faces_in_image(
    file: UploadFile = File(..., description="Image file"),
    request: DetectionRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Detect and/or recognize faces in uploaded image"""
    
    try:
        # Validate file
        if not validate_image(file.filename):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Validate file size
        await validate_file_size(file)
        
        # Generate unique ID
        image_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_path = Path(settings.upload_dir) / f"{image_id}_{file.filename}"
        
        async with aiofiles.open(upload_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process image
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(upload_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Cannot read image file")
        
        # Get services
        face_detector = get_face_detector()
        
        # Detect faces
        detected_faces = face_detector.detect_faces(
            image, 
            conf_threshold=request.confidence_threshold
        )
        
        faces_result = []
        faces_recognized = 0
        
        # Process each detected face
        for i, face_data in enumerate(detected_faces):
            bbox_data = face_data['bbox']
            bbox = BoundingBox(
                x1=bbox_data['x1'],
                y1=bbox_data['y1'],
                x2=bbox_data['x2'],
                y2=bbox_data['y2']
            )
            
            face_detection = FaceDetection(
                bbox=bbox,
                confidence=face_data['confidence'],
                face_id=f"{image_id}_face_{i}"
            )
            
            recognition_result = None
            
            # Recognize face if requested
            if request.processing_mode in [ProcessingMode.RECOGNIZE_ONLY, ProcessingMode.DETECT_AND_RECOGNIZE]:
                face_recognizer = get_face_recognizer()
                recog_result = face_recognizer.recognize_face(face_data['face_crop'])
                
                if recog_result:
                    recognition_result = FaceRecognition(
                        person_name=recog_result['person_name'],
                        confidence=recog_result['confidence'],
                        class_id=recog_result['class_id']
                    )
                    faces_recognized += 1
            
            processed_face = ProcessedFace(
                detection=face_detection,
                recognition=recognition_result
            )
            
            faces_result.append(processed_face)
        
        # Save processed image with annotations
        output_path = None
        if faces_result:
            output_path = save_processed_image(image, faces_result, image_id)
        
        processing_time = time.time() - start_time
        
        # Create response
        result = ImageProcessingResult(
            image_id=image_id,
            filename=file.filename,
            processing_time=processing_time,
            faces_detected=len(detected_faces),
            faces_recognized=faces_recognized,
            faces=faces_result,
            output_image_path=output_path,
            timestamp=datetime.now()
        )
        
        # Cleanup original file in background
        background_tasks.add_task(cleanup_file, upload_path)
        
        logger.info(f"Processed image {image_id}: {len(detected_faces)} faces detected, {faces_recognized} recognized")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/detect/video", response_model=VideoProcessingResult)
async def detect_faces_in_video(
    file: UploadFile = File(..., description="Video file"),
    request: VideoProcessingRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Detect and/or recognize faces in uploaded video"""
    
    try:
        # Validate file
        if not validate_video(file.filename):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        # Generate unique ID
        video_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_path = Path(settings.upload_dir) / f"{video_id}_{file.filename}"
        output_path = Path(settings.output_dir) / f"{video_id}_processed.mp4"
        
        async with aiofiles.open(upload_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process video
        start_time = time.time()
        video_processor = get_video_processor()
        
        result_data = video_processor.process_video_file(
            video_path=str(upload_path),
            output_path=str(output_path),
            confidence_threshold=request.confidence_threshold,
            frame_skip=request.frame_skip,
            max_frames=request.max_frames
        )
        
        processing_time = time.time() - start_time
        
        # Create response
        result = VideoProcessingResult(
            video_id=video_id,
            filename=file.filename,
            total_frames=result_data['total_frames'],
            processed_frames=result_data['processed_frames'],
            processing_time=processing_time,
            faces_detected=result_data['faces_detected'],
            faces_recognized=result_data['faces_recognized'],
            output_video_path=str(output_path) if output_path.exists() else None,
            frame_results=result_data['frame_results'],
            timestamp=datetime.now()
        )
        
        # Cleanup original file in background
        background_tasks.add_task(cleanup_file, upload_path)
        
        logger.info(f"Processed video {video_id}: {result_data['processed_frames']} frames")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/download/image/{image_id}")
async def download_processed_image(image_id: str):
    """Download processed image with annotations"""
    
    output_path = Path(settings.output_dir) / f"{image_id}_processed.jpg"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    return FileResponse(
        path=output_path,
        media_type="image/jpeg",
        filename=f"{image_id}_processed.jpg"
    )

@router.get("/download/video/{video_id}")
async def download_processed_video(video_id: str):
    """Download processed video with annotations"""
    
    output_path = Path(settings.output_dir) / f"{video_id}_processed.mp4"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"{video_id}_processed.mp4"
    )

@router.get("/stream/video/{video_id}")
async def stream_video_processing(video_id: str):
    """Stream video processing results in real-time"""
    
    upload_path = Path(settings.upload_dir) / f"{video_id}.mp4"
    
    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    def generate_stream():
        video_processor = get_video_processor()
        try:
            for frame_data in video_processor.process_video_stream(str(upload_path)):
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       frame_data['frame_data'] + b'\r\n')
        except Exception as e:
            logger.error(f"Error streaming video: {e}")
    
    return StreamingResponse(
        generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

async def cleanup_file(file_path: Path):
    """Background task to cleanup temporary files"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")