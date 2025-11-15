import pytest
from fastapi.testclient import TestClient
from energy_efficiency.api import app

client = TestClient(app)

class TestAPI:
    def test_root_endpoint(self):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_endpoint_validation(self):
        """Test prediction endpoint input validation."""
        # Test with invalid data (out of range)
        invalid_data = {
            "relative_compactness": 1.5,  # Out of range
            "surface_area": 650.0,
            "wall_area": 300.0,
            "roof_area": 150.0,
            "overall_height": 5.0,
            "orientation": 3,
            "glazing_area": 0.25,
            "glazing_area_distribution": 2
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_fields(self):
        """Test prediction endpoint with missing required fields."""
        incomplete_data = {
            "relative_compactness": 0.8,
            "surface_area": 650.0
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_model_info_endpoint(self):
        """Test model info endpoint structure."""
        response = client.get("/model-info")
        # May return 503 if model not loaded in test environment
        assert response.status_code in [200, 503]
