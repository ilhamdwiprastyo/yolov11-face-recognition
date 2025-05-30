import pytest
import requests
import io
from PIL import Image
import numpy as np

# Test configuration
API_BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/v1/health"
IMAGE_ENDPOINT = f"{API_BASE_URL}/api/v1/detect/image"

def create_test_image():
    """Create a test image for testing"""
    # Create a simple test image
    img = Image.new('RGB', (640, 480), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

class TestAPI:
    
    def test_health_check(self):
        """Test API health endpoint"""
        response = requests.get(HEALTH_ENDPOINT)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "gpu_available" in data
    
    def test_image_upload_success(self):
        """Test successful image upload"""
        test_image = create_test_image()
        
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        data = {
            "confidence_threshold": 0.5,
            "processing_mode": "detect_and_recognize"
        }
        
        response = requests.post(IMAGE_ENDPOINT, files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "image_id" in result
        assert "faces_detected" in result
        assert "processing_time" in result
    
    def test_image_upload_invalid_format(self):
        """Test upload with invalid image format"""
        # Create a text file instead of image
        test_file = io.BytesIO(b"This is not an image")
        
        files = {"file": ("test.txt", test_file, "text/plain")}
        data = {"confidence_threshold": 0.5}
        
        response = requests.post(IMAGE_ENDPOINT, files=files, data=data)
        assert response.status_code == 400
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation"""
        test_image = create_test_image()
        
        # Test invalid confidence threshold
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        data = {"confidence_threshold": 1.5}  # Invalid: > 1.0
        
        response = requests.post(IMAGE_ENDPOINT, files=files, data=data)
        assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    pytest.main([__file__])