import cv2
import numpy as np
from typing import List, Dict, Any, Generator, Optional
from pathlib import Path
import time
from app.services.face_detection import get_face_detector
from app.services.face_recognition import get_face_recognizer
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

class VideoProcessor:
    def __init__(self):
        self.face_detector = get_face_detector()
        self.face_recognizer = get_face_recognizer()
        
    def process_video_file(self, 
                          video_path: str,
                          output_path: str = None,
                          confidence_threshold: float = 0.5,
                          frame_skip: int = 1,
                          max_frames: int = None,
                          return_embeddings: bool = False) -> Dict[str, Any]:
        """Process entire video file"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Setup output video if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing variables
        frame_results = []
        processed_frames = 0
        total_faces_detected = 0
        total_faces_recognized = 0
        start_time = time.time()
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                # Check max frames limit
                if max_frames and processed_frames >= max_frames:
                    break
                
                # Process frame
                frame_result = self.process_frame(
                    frame,
                    confidence_threshold=confidence_threshold,
                    return_embeddings=return_embeddings,
                    frame_number=frame_number
                )
                
                frame_results.append(frame_result)
                total_faces_detected += frame_result['faces_detected']
                total_faces_recognized += frame_result['faces_recognized']
                
                # Draw results on frame
                annotated_frame = self.draw_results_on_frame(frame, frame_result['faces'])
                
                # Write to output video
                if out:
                    out.write(annotated_frame)
                
                processed_frames += 1
                frame_number += 1
                
                # Log progress
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames...")
                    
        finally:
            cap.release()
            if out:
                out.release()
        
        processing_time = time.time() - start_time
        
        result = {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'processing_time': processing_time,
            'faces_detected': total_faces_detected,
            'faces_recognized': total_faces_recognized,
            'frame_results': frame_results,
            'output_video_path': output_path
        }
        
        logger.info(f"Video processing completed: {processed_frames} frames in {processing_time:.2f}s")
        
        return result
    
    def process_frame(self,
                     frame: np.ndarray,
                     confidence_threshold: float = 0.5,
                     return_embeddings: bool = False,
                     frame_number: int = 0) -> Dict[str, Any]:
        """Process a single frame"""
        
        # Detect faces
        detected_faces = self.face_detector.detect_faces(frame, confidence_threshold)
        
        # Recognize faces
        faces_result = []
        faces_recognized = 0
        
        for face_data in detected_faces:
            face_crop = face_data['face_crop']
            bbox = face_data['bbox']
            detection_conf = face_data['confidence']
            
            # Recognize face
            recognition_result = self.face_recognizer.recognize_face(
                face_crop, 
                return_embedding=return_embeddings
            )
            
            face_result = {
                'detection': {
                    'bbox': bbox,
                    'confidence': detection_conf,
                    'face_id': f"frame_{frame_number}_face_{len(faces_result)}"
                },
                'recognition': recognition_result
            }
            
            if recognition_result:
                faces_recognized += 1
            
            faces_result.append(face_result)
        
        return {
            'frame_number': frame_number,
            'faces_detected': len(detected_faces),
            'faces_recognized': faces_recognized,
            'faces': faces_result
        }
    
    def draw_results_on_frame(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection and recognition results on frame"""
        annotated_frame = frame.copy()
        
        for face in faces:
            bbox = face['detection']['bbox']
            detection_conf = face['detection']['confidence']
            recognition = face['recognition']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         (0, 255, 0), 2)
            
            # Prepare text
            if recognition and recognition['confidence'] > 0.5:
                text = f"{recognition['person_name']}: {recognition['confidence']:.2f}"
                color = (0, 255, 0)  # Green for recognized
            else:
                text = f"Unknown: {detection_conf:.2f}"
                color = (0, 0, 255)  # Red for unknown
            
            # Draw text background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(annotated_frame,
                         (bbox['x1'], bbox['y1'] - text_height - 10),
                         (bbox['x1'] + text_width, bbox['y1']),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, text,
                       (bbox['x1'], bbox['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video_stream(self, 
                           video_path: str,
                           confidence_threshold: float = 0.5) -> Generator[Dict[str, Any], None, None]:
        """Process video as a stream for real-time applications"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_result = self.process_frame(
                    frame,
                    confidence_threshold=confidence_threshold,
                    frame_number=frame_number
                )
                
                # Draw results
                annotated_frame = self.draw_results_on_frame(frame, frame_result['faces'])
                
                # Encode frame for streaming
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                
                yield {
                    'frame_number': frame_number,
                    'frame_data': frame_bytes,
                    'results': frame_result
                }
                
                frame_number += 1
                
        finally:
            cap.release()

    def extract_frames(self, 
                      video_path: str, 
                      output_dir: str, 
                      frame_skip: int = 30) -> List[str]:
        """Extract frames from video for analysis"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        frame_number = 0
        saved_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame if it's the right interval
                if frame_number % frame_skip == 0:
                    frame_filename = f"frame_{saved_frames:06d}.jpg"
                    frame_path = output_path / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    saved_frames += 1
                
                frame_number += 1
                
        finally:
            cap.release()
        
        logger.info(f"Extracted {saved_frames} frames from video")
        return frame_paths
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            return {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'resolution': f"{width}x{height}",
                'file_path': video_path
            }
            
        finally:
            cap.release()
    
    def create_video_summary(self, 
                           video_path: str, 
                           sample_frames: int = 10) -> Dict[str, Any]:
        """Create a summary of faces detected throughout the video"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // sample_frames)
        
        face_statistics = {
            'total_faces_detected': 0,
            'unique_persons': set(),
            'frames_with_faces': 0,
            'average_faces_per_frame': 0,
            'confidence_distribution': [],
            'person_appearances': {}
        }
        
        frames_processed = 0
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only sample frames
                if frame_number % frame_interval == 0:
                    frame_result = self.process_frame(frame, frame_number=frame_number)
                    
                    if frame_result['faces_detected'] > 0:
                        face_statistics['frames_with_faces'] += 1
                        face_statistics['total_faces_detected'] += frame_result['faces_detected']
                        
                        for face in frame_result['faces']:
                            if face['recognition']:
                                person_name = face['recognition']['person_name']
                                confidence = face['recognition']['confidence']
                                
                                face_statistics['unique_persons'].add(person_name)
                                face_statistics['confidence_distribution'].append(confidence)
                                
                                if person_name not in face_statistics['person_appearances']:
                                    face_statistics['person_appearances'][person_name] = 0
                                face_statistics['person_appearances'][person_name] += 1
                    
                    frames_processed += 1
                
                frame_number += 1
                
        finally:
            cap.release()
        
        # Calculate final statistics
        if frames_processed > 0:
            face_statistics['average_faces_per_frame'] = face_statistics['total_faces_detected'] / frames_processed
        
        face_statistics['unique_persons'] = list(face_statistics['unique_persons'])
        
        if face_statistics['confidence_distribution']:
            face_statistics['average_confidence'] = sum(face_statistics['confidence_distribution']) / len(face_statistics['confidence_distribution'])
        else:
            face_statistics['average_confidence'] = 0
        
        return {
            'video_info': self.get_video_info(video_path),
            'face_statistics': face_statistics,
            'frames_processed': frames_processed
        }

# Global video processor instance
video_processor = None

def get_video_processor() -> VideoProcessor:
    """Get global video processor instance"""
    global video_processor
    if video_processor is None:
        video_processor = VideoProcessor()
    return video_processor