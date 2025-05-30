import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import time
import io
import os
import base64
from typing import Optional, Dict, Any, List
import json
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Streamlit configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'api_url' not in st.session_state:
        st.session_state.api_url = None
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = None
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'current_file_type' not in st.session_state:
        st.session_state.current_file_type = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = 'idle'
    if 'uploaded_file_info' not in st.session_state:
        st.session_state.uploaded_file_info = None
    if 'frame_player_index' not in st.session_state:
        st.session_state.frame_player_index = 0
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = False

# API Configuration
def get_api_url():
    """Get API URL with proper fallback logic"""
    env_api_url = os.getenv('API_URL', '').strip()
    if env_api_url:
        return env_api_url
    
    fallback_urls = [
        "http://face-recognition-api:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
    
    return fallback_urls

def test_api_connection(api_url: str, timeout: int = 5) -> bool:
    """Test single API URL"""
    try:
        response = requests.get(f"{api_url}/api/v1/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

def find_working_api_url() -> Optional[str]:
    """Find working API URL from available options"""
    api_urls = get_api_url()
    
    if isinstance(api_urls, str):
        if test_api_connection(api_urls):
            return api_urls
        else:
            return None
    
    for url in api_urls:
        if test_api_connection(url):
            return url
    
    return None

@st.cache_data(ttl=60)
def check_api_health(api_url: str) -> Dict[str, Any]:
    """Check API health with caching"""
    try:
        response = requests.get(f"{api_url}/api/v1/health", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def upload_image(api_url: str, file, confidence_threshold: float = 0.5) -> Optional[Dict[str, Any]]:
    """Upload image to API for processing"""
    try:
        files = {"file": file}
        data = {
            "confidence_threshold": confidence_threshold,
            "processing_mode": "detect_and_recognize"
        }
        
        response = requests.post(
            f"{api_url}/api/v1/detect/image",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error uploading image: {e}")
        return None

def upload_video(api_url: str, file, confidence_threshold: float = 0.5, frame_skip: int = 1, max_frames: int = 100) -> Optional[Dict[str, Any]]:
    """Upload video to API for processing"""
    try:
        files = {"file": file}
        data = {
            "confidence_threshold": confidence_threshold,
            "processing_mode": "detect_and_recognize",
            "frame_skip": frame_skip,
            "max_frames": max_frames
        }
        
        response = requests.post(
            f"{api_url}/api/v1/detect/video",
            files=files,
            data=data,
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error uploading video: {e}")
        return None

def download_processed_video(api_url: str, video_id: str) -> Optional[bytes]:
    """Download processed video from API"""
    try:
        response = requests.get(f"{api_url}/api/v1/download/video/{video_id}", timeout=60)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Download Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

def create_frame_analysis_chart(frame_results: List[Dict[str, Any]]) -> None:
    """Create interactive chart for frame-by-frame analysis"""
    
    if not frame_results:
        st.warning("No frame data available for analysis")
        return
    
    # Prepare data
    frames = []
    faces_detected = []
    faces_recognized = []
    recognition_rates = []
    
    for frame in frame_results:
        frames.append(frame['frame_number'])
        detected = frame['faces_detected']
        recognized = frame['faces_recognized']
        
        faces_detected.append(detected)
        faces_recognized.append(recognized)
        
        # Calculate recognition rate
        rate = (recognized / detected * 100) if detected > 0 else 0
        recognition_rates.append(rate)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Frame': frames,
        'Faces Detected': faces_detected,
        'Faces Recognized': faces_recognized,
        'Recognition Rate (%)': recognition_rates
    })
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Faces Detection Over Time")
        fig1 = px.line(df, x='Frame', y=['Faces Detected', 'Faces Recognized'],
                      title="Face Detection & Recognition Timeline",
                      labels={'value': 'Number of Faces', 'variable': 'Type'})
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📈 Recognition Rate")
        fig2 = px.bar(df, x='Frame', y='Recognition Rate (%)',
                     title="Recognition Success Rate per Frame",
                     color='Recognition Rate (%)',
                     color_continuous_scale='RdYlGn')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Summary statistics
    st.subheader("📋 Video Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_faces = df['Faces Detected'].mean()
        st.metric("Avg Faces/Frame", f"{avg_faces:.1f}")
    
    with col2:
        total_detections = df['Faces Detected'].sum()
        st.metric("Total Detections", total_detections)
    
    with col3:
        avg_recognition_rate = df['Recognition Rate (%)'].mean()
        st.metric("Avg Recognition Rate", f"{avg_recognition_rate:.1f}%")
    
    with col4:
        frames_with_faces = len(df[df['Faces Detected'] > 0])
        st.metric("Frames with Faces", f"{frames_with_faces}/{len(df)}")

def display_person_appearances(frame_results: List[Dict[str, Any]]) -> None:
    """Display person appearances throughout the video"""
    
    person_appearances = {}
    person_confidences = {}
    
    for frame in frame_results:
        for face in frame.get('faces', []):
            recognition = face.get('recognition')
            if recognition:
                person_name = recognition['person_name']
                confidence = recognition['confidence']
                
                if person_name not in person_appearances:
                    person_appearances[person_name] = 0
                    person_confidences[person_name] = []
                
                person_appearances[person_name] += 1
                person_confidences[person_name].append(confidence)
    
    if person_appearances:
        st.subheader("👥 Person Appearances in Video")
        
        # Create DataFrame for visualization
        persons = list(person_appearances.keys())
        appearances = list(person_appearances.values())
        avg_confidences = [sum(person_confidences[person])/len(person_confidences[person]) 
                          for person in persons]
        
        df_persons = pd.DataFrame({
            'Person': persons,
            'Appearances': appearances,
            'Avg Confidence': avg_confidences
        })
        
        # Sort by appearances
        df_persons = df_persons.sort_values('Appearances', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of appearances
            fig = px.bar(df_persons, x='Person', y='Appearances',
                        title="Person Appearances Count",
                        color='Avg Confidence',
                        color_continuous_scale='Viridis')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display detailed table
            st.write("**Detailed Statistics:**")
            for _, row in df_persons.iterrows():
                with st.expander(f"👤 {row['Person']} ({row['Appearances']} appearances)"):
                    st.write(f"**Average Confidence:** {row['Avg Confidence']:.3f}")
                    confidences = person_confidences[row['Person']]
                    st.write(f"**Min Confidence:** {min(confidences):.3f}")
                    st.write(f"**Max Confidence:** {max(confidences):.3f}")

def display_video_player(video_bytes: bytes, video_name: str = "processed_video.mp4"):
    """Display video player in Streamlit - Improved version"""
    try:
        st.subheader("🎥 Processed Video with Face Recognition")
        
        # Display video directly using st.video
        st.video(video_bytes)
        
        # Video info
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"📹 **Video:** {video_name}")
        with col2:
            st.info(f"📁 **Size:** {len(video_bytes) / (1024*1024):.1f} MB")
        
        # Download button
        st.download_button(
            label="📥 Download Processed Video",
            data=video_bytes,
            file_name=video_name,
            mime="video/mp4",
            type="primary"
        )
        
    except Exception as e:
        st.error(f"Error displaying video: {e}")

def display_results(results: Dict[str, Any], file_type: str = "image"):
    """Display processing results - Simplified version for better video display"""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Faces Detected", results["faces_detected"])
    
    with col2:
        st.metric("Faces Recognized", results["faces_recognized"])
    
    with col3:
        st.metric("Processing Time", f"{results['processing_time']:.2f}s")
    
    if file_type == "video":
        with col4:
            st.metric("Frames Processed", f"{results['processed_frames']}/{results['total_frames']}")
    
    # Handle different structures for image vs video
    if file_type == "image":
        # Image processing display (tetap sama)
        faces_to_display = results.get("faces", [])
        
        st.subheader("🔍 Detection Details")
        if faces_to_display:
            for i, face in enumerate(faces_to_display):
                with st.expander(f"Face {i+1}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Detection:**")
                        st.write(f"- Confidence: {face['detection']['confidence']:.3f}")
                        bbox = face['detection']['bbox']
                        st.write(f"- Bounding Box: ({bbox['x1']}, {bbox['y1']}) - ({bbox['x2']}, {bbox['y2']})")
                    
                    with col2:
                        if face["recognition"]:
                            st.write("**Recognition:**")
                            st.write(f"- Person: **{face['recognition']['person_name']}**")
                            st.write(f"- Confidence: {face['recognition']['confidence']:.3f}")
                            st.write(f"- Class ID: {face['recognition']['class_id']}")
                        else:
                            st.write("**Recognition:** Not recognized")
        else:
            st.info("No faces detected in the uploaded image.")
    
    elif file_type == "video":
        # Video processing - SIMPLIFIED VERSION
        frame_results = results.get("frame_results", [])
        
        if frame_results:
            # Show video analysis charts
            create_frame_analysis_chart(frame_results)
            
            # Show person appearances
            display_person_appearances(frame_results)
            
            # Main video display section - IMPROVED
            st.markdown("---")
            if results.get("output_video_path"):
                
                # Try to download and display video immediately
                if st.button("🎥 Load Processed Video", type="primary"):
                    with st.spinner("Loading processed video..."):
                        video_id = results["video_id"]
                        video_bytes = download_processed_video(st.session_state.api_url, video_id)
                        
                        if video_bytes:
                            st.success("✅ Video loaded successfully!")
                            display_video_player(
                                video_bytes, 
                                f"{results['filename']}_processed.mp4"
                            )
                        else:
                            st.error("❌ Failed to load processed video")
                
                # Alternative: Show detailed frame analysis (optional)
                with st.expander("📊 Advanced Frame Analysis", expanded=False):
                    display_detailed_frame_stats(frame_results)
            
        else:
            st.info("No frame results available.")

def display_detailed_frame_stats(frame_results: List[Dict[str, Any]]) -> None:
    """Display detailed frame statistics - optional advanced view"""
    
    st.subheader("📋 Frame-by-Frame Statistics")
    
    # Create a simple table view instead of complex player
    df_frames = []
    for frame in frame_results:
        row = {
            'Frame': frame['frame_number'],
            'Faces Detected': frame['faces_detected'],
            'Faces Recognized': frame['faces_recognized'],
            'Recognition Rate': f"{(frame['faces_recognized']/frame['faces_detected']*100) if frame['faces_detected'] > 0 else 0:.1f}%"
        }
        
        # Add person names if available
        persons = []
        for face in frame.get('faces', []):
            if face.get('recognition'):
                persons.append(face['recognition']['person_name'])
        row['Recognized Persons'] = ', '.join(persons) if persons else 'None'
        
        df_frames.append(row)
    
    # Display as dataframe (much more efficient than custom player)
    df = pd.DataFrame(df_frames)
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        min_faces = st.slider("Min faces detected", 0, df['Faces Detected'].max(), 0)
    with col2:
        show_only_recognized = st.checkbox("Show only frames with recognized faces")
    
    # Filter data
    filtered_df = df[df['Faces Detected'] >= min_faces]
    if show_only_recognized:
        filtered_df = filtered_df[filtered_df['Faces Recognized'] > 0]
    
    # Display filtered results
    st.dataframe(
        filtered_df, 
        use_container_width=True,
        height=400
    )
    
    # Summary for filtered data
    if len(filtered_df) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frames Shown", len(filtered_df))
        with col2:
            avg_faces = filtered_df['Faces Detected'].mean()
            st.metric("Avg Faces/Frame", f"{avg_faces:.1f}")
        with col3:
            total_faces = filtered_df['Faces Detected'].sum()
            st.metric("Total Faces", total_faces)

def reset_processing_state():
    """Reset processing-related session state"""
    st.session_state.current_results = None
    st.session_state.current_file_type = None
    st.session_state.processing_status = 'idle'
    st.session_state.uploaded_file_info = None
    st.session_state.frame_player_index = 0
    st.session_state.auto_play = False
    
    if 'processed_video_bytes' in st.session_state:
        del st.session_state.processed_video_bytes
    if 'processed_video_name' in st.session_state:
        del st.session_state.processed_video_name
    if 'temp_video_path' in st.session_state:
        try:
            if os.path.exists(st.session_state.temp_video_path):
                os.unlink(st.session_state.temp_video_path)
        except:
            pass
        del st.session_state.temp_video_path

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    st.title("🔍 Face Recognition System")
    st.markdown("Upload images or videos to detect and recognize faces using advanced AI models.")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # API Connection Section
    st.sidebar.subheader("🔗 API Connection")
    
    # Find API URL if not already found
    if st.session_state.api_url is None:
        with st.spinner("🔍 Searching for API server..."):
            api_url = find_working_api_url()
            
            if api_url:
                st.session_state.api_url = api_url
                st.session_state.connection_status = "connected"
                st.sidebar.success(f"✅ Connected to: {api_url}")
            else:
                st.session_state.connection_status = "failed"
                st.sidebar.error("❌ API connection failed")
    
    # Connection status display
    if st.session_state.connection_status == "connected":
        health_data = check_api_health(st.session_state.api_url)
        if health_data.get("status") == "healthy":
            st.sidebar.success("✅ API is healthy")
        else:
            st.sidebar.warning(f"⚠️ API status: {health_data.get('status', 'unknown')}")
        
        st.sidebar.info(f"🌐 API: {st.session_state.api_url}")
        
        if st.sidebar.button("🔄 Refresh Connection"):
            st.session_state.api_url = None
            st.session_state.connection_status = None
            reset_processing_state()
            st.rerun()
    
    elif st.session_state.connection_status == "failed":
        st.sidebar.error("❌ Cannot connect to API")
        
        if st.sidebar.button("🔄 Retry Connection"):
            st.session_state.api_url = None
            st.session_state.connection_status = None
            st.rerun()
        
        st.error("Cannot connect to the Face Recognition API.")
        st.markdown("""
        **Troubleshooting Steps:**
        1. Make sure API container is running: `docker ps`
        2. Check API health manually: `curl http://localhost:8000/api/v1/health`
        3. Check container logs: `docker logs face-recognition-api`
        4. Restart services: `docker-compose restart`
        """)
        st.stop()
    
    # If connected, show main interface
    if st.session_state.connection_status == "connected":
        # Reset button in sidebar
        if st.sidebar.button("🗑️ Clear Results", help="Clear current processing results"):
            reset_processing_state()
            st.rerun()
        
        # Processing settings
        confidence_threshold = st.sidebar.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for face detection"
        )
        
        # File type selection
        file_type = st.sidebar.selectbox(
            "File Type",
            ["Image", "Video"],
            help="Select the type of file you want to process"
        )
        
        # Show current processing status
        if st.session_state.processing_status != 'idle':
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Processing Status")
            if st.session_state.processing_status == 'completed':
                st.sidebar.success("✅ Processing completed")
            elif st.session_state.processing_status == 'processing':
                st.sidebar.info("⏳ Processing in progress...")
            elif st.session_state.processing_status == 'error':
                st.sidebar.error("❌ Processing failed")
        
        # Main content based on processing status
        if st.session_state.processing_status == 'completed' and st.session_state.current_results:
            # Show results
            st.header(f"📊 {st.session_state.current_file_type.title()} Processing Results")
            
            # Show file info
            if st.session_state.uploaded_file_info:
                with st.expander("📁 File Information", expanded=False):
                    info = st.session_state.uploaded_file_info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Filename:** {info['name']}")
                    with col2:
                        st.write(f"**Size:** {info['size']}")
                    with col3:
                        st.write(f"**Type:** {info['type']}")
            
            # Display results
            display_results(st.session_state.current_results, st.session_state.current_file_type)
            
        else:
            # Show upload interface
            if file_type == "Image":
                st.header("📸 Image Processing")
                
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=["jpg", "jpeg", "png", "bmp"],
                    help="Upload an image file to detect and recognize faces"
                )
                
                if uploaded_file is not None:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Original Image")
                        image = Image.open(uploaded_file)
                        st.image(image, use_column_width=True)
                        
                        st.write(f"**Filename:** {uploaded_file.name}")
                        st.write(f"**Size:** {image.size}")
                        st.write(f"**Format:** {image.format}")
                    
                    with col2:
                        st.subheader("Processing")
                        
                        if st.button("🚀 Process Image", type="primary"):
                            st.session_state.processing_status = 'processing'
                            
                            # Store file info
                            st.session_state.uploaded_file_info = {
                                'name': uploaded_file.name,
                                'size': f"{image.size[0]}x{image.size[1]}",
                                'type': image.format
                            }
                            
                            with st.spinner("Processing image... This may take a few seconds."):
                                uploaded_file.seek(0)
                                results = upload_image(st.session_state.api_url, uploaded_file, confidence_threshold)
                                
                                if results:
                                    st.session_state.current_results = results
                                    st.session_state.current_file_type = "image"
                                    st.session_state.processing_status = 'completed'
                                    st.rerun()
                                else:
                                    st.session_state.processing_status = 'error'
                                    st.error("❌ Failed to process image")
            
            else:  # Video processing
                st.header("🎥 Video Processing with Frame Player")
                
                # Video settings
                col1, col2 = st.columns(2)
                
                with col1:
                    frame_skip = st.slider(
                        "Frame Skip",
                        min_value=1,
                        max_value=10,
                        value=2,
                        help="Process every Nth frame to speed up processing"
                    )
                
                with col2:
                    max_frames = st.slider(
                        "Max Frames",
                        min_value=10,
                        max_value=500,
                        value=100,
                        help="Maximum number of frames to process"
                    )
                
                uploaded_file = st.file_uploader(
                    "Choose a video file",
                    type=["mp4", "avi", "mov", "mkv"],
                    help="Upload a video file to detect and recognize faces"
                )
                
                if uploaded_file is not None:
                    st.subheader("Video Information")
                    st.write(f"**Filename:** {uploaded_file.name}")
                    st.write(f"**Size:** {uploaded_file.size / (1024*1024):.1f} MB")
                    
                    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
                        st.warning("⚠️ Large video file detected. Processing may take several minutes.")
                    
                    if st.button("🚀 Process Video", type="primary"):
                        st.session_state.processing_status = 'processing'
                        
                        # Store file info
                        st.session_state.uploaded_file_info = {
                            'name': uploaded_file.name,
                            'size': f"{uploaded_file.size / (1024*1024):.1f} MB",
                            'type': 'Video'
                        }
                        
                        with st.spinner("Processing video... This may take several minutes."):
                            uploaded_file.seek(0)
                            results = upload_video(
                                st.session_state.api_url,
                                uploaded_file,
                                confidence_threshold,
                                frame_skip,
                                max_frames
                            )
                            
                            if results:
                                st.session_state.current_results = results
                                st.session_state.current_file_type = "video"
                                st.session_state.processing_status = 'completed'
                                st.rerun()
                            else:
                                st.session_state.processing_status = 'error'
                                st.error("❌ Failed to process video")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Powered by YOLOv11 Face Detection and ArcFace Recognition</p>
            <p>Built with ❤️ using FastAPI and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()