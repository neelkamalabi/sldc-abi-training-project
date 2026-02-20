"""
Test suite for FastAPI Iris Classifier API.
"""
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from iris_api.app import BatchIrisFeatures, IrisFeatures, app


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
    return model


@pytest.fixture
def mock_artifacts(tmp_path, mock_model):
    """Create mock artifacts directory with model files."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    
    # Save mock model
    joblib.dump(mock_model, artifacts_dir / "iris_classifier.joblib")
    
    # Save feature names
    feature_names = ["sepal length (cm)", "sepal width (cm)", 
                    "petal length (cm)", "petal width (cm)"]
    joblib.dump(feature_names, artifacts_dir / "feature_names.joblib")
    
    # Save target names
    target_names = ["setosa", "versicolor", "virginica"]
    joblib.dump(target_names, artifacts_dir / "target_names.joblib")
    
    return artifacts_dir


@pytest.fixture
def client(mock_artifacts):
    """Create a test client with mocked model loading."""
    with patch('iris_api.app.Path') as mock_path:
        # Mock the path to artifacts directory
        mock_file = Mock()
        mock_file.parent.parent.parent = mock_artifacts.parent
        mock_path.return_value = mock_file
        
        # Create test client and trigger startup
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def sample_iris_features():
    """Sample iris features for testing."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Iris Classifier API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
        assert "health" in data["endpoints"]
        assert "predict" in data["endpoints"]
        assert "batch_predict" in data["endpoints"]


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "features" in data
        assert "classes" in data
        assert len(data["classes"]) == 3
    
    def test_health_check_returns_feature_names(self, client):
        """Test health check returns feature names."""
        response = client.get("/health")
        data = response.json()
        
        assert len(data["features"]) == 4
        assert "sepal length (cm)" in data["features"]
        assert "petal width (cm)" in data["features"]
    
    def test_health_check_returns_target_names(self, client):
        """Test health check returns target class names."""
        response = client.get("/health")
        data = response.json()
        
        assert "setosa" in data["classes"]
        assert "versicolor" in data["classes"]
        assert "virginica" in data["classes"]


class TestPredictEndpoint:
    """Tests for single prediction endpoint."""
    
    def test_predict_success(self, client, sample_iris_features):
        """Test successful prediction."""
        response = client.post("/predict", json=sample_iris_features)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_class" in data
        assert "predicted_species" in data
        assert "confidence" in data
        assert "probabilities" in data
        
        # Verify data types
        assert isinstance(data["predicted_class"], int)
        assert isinstance(data["predicted_species"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["probabilities"], dict)
    
    def test_predict_returns_valid_class(self, client, sample_iris_features):
        """Test prediction returns valid class index."""
        response = client.post("/predict", json=sample_iris_features)
        data = response.json()
        
        assert 0 <= data["predicted_class"] <= 2
    
    def test_predict_returns_valid_species(self, client, sample_iris_features):
        """Test prediction returns valid species name."""
        response = client.post("/predict", json=sample_iris_features)
        data = response.json()
        
        assert data["predicted_species"] in ["setosa", "versicolor", "virginica"]
    
    def test_predict_confidence_in_range(self, client, sample_iris_features):
        """Test confidence is between 0 and 1."""
        response = client.post("/predict", json=sample_iris_features)
        data = response.json()
        
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_predict_probabilities_sum_to_one(self, client, sample_iris_features):
        """Test probabilities sum to approximately 1."""
        response = client.post("/predict", json=sample_iris_features)
        data = response.json()
        
        prob_sum = sum(data["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_predict_probabilities_for_all_classes(self, client, sample_iris_features):
        """Test probabilities returned for all three classes."""
        response = client.post("/predict", json=sample_iris_features)
        data = response.json()
        
        assert len(data["probabilities"]) == 3
        assert "setosa" in data["probabilities"]
        assert "versicolor" in data["probabilities"]
        assert "virginica" in data["probabilities"]
    
    def test_predict_missing_field(self, client):
        """Test prediction fails with missing required field."""
        incomplete_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4
            # Missing petal_width
        }
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    def test_predict_negative_value(self, client):
        """Test prediction fails with negative value."""
        invalid_data = {
            "sepal_length": -5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_invalid_data_type(self, client):
        """Test prediction fails with invalid data type."""
        invalid_data = {
            "sepal_length": "not_a_number",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_extreme_values(self, client):
        """Test prediction with extreme but valid values."""
        extreme_data = {
            "sepal_length": 10.0,
            "sepal_width": 8.0,
            "petal_length": 7.0,
            "petal_width": 3.0
        }
        response = client.post("/predict", json=extreme_data)
        assert response.status_code == 200


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""
    
    def test_batch_predict_success(self, client, sample_iris_features):
        """Test successful batch prediction."""
        batch_data = {
            "samples": [
                sample_iris_features,
                {
                    "sepal_length": 6.5,
                    "sepal_width": 3.0,
                    "petal_length": 5.2,
                    "petal_width": 2.0
                }
            ]
        }
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    def test_batch_predict_single_sample(self, client, sample_iris_features):
        """Test batch prediction with single sample."""
        batch_data = {"samples": [sample_iris_features]}
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == 1
    
    def test_batch_predict_multiple_samples(self, client):
        """Test batch prediction with multiple samples."""
        batch_data = {
            "samples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, 
                 "petal_length": 1.4, "petal_width": 0.2},
                {"sepal_length": 6.5, "sepal_width": 3.0, 
                 "petal_length": 5.2, "petal_width": 2.0},
                {"sepal_length": 5.9, "sepal_width": 3.0, 
                 "petal_length": 4.2, "petal_width": 1.5}
            ]
        }
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == 3
        
        # Verify each prediction has required fields
        for pred in data["predictions"]:
            assert "predicted_class" in pred
            assert "predicted_species" in pred
            assert "confidence" in pred
            assert "probabilities" in pred
    
    def test_batch_predict_empty_samples(self, client):
        """Test batch prediction fails with empty samples list."""
        batch_data = {"samples": []}
        response = client.post("/predict/batch", json=batch_data)
        # Should still return 200 with empty predictions
        assert response.status_code == 200
    
    def test_batch_predict_invalid_sample(self, client, sample_iris_features):
        """Test batch prediction fails with invalid sample in batch."""
        batch_data = {
            "samples": [
                sample_iris_features,
                {"sepal_length": -5.1, "sepal_width": 3.5,  # Negative value
                 "petal_length": 1.4, "petal_width": 0.2}
            ]
        }
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 422


class TestIrisFeaturesModel:
    """Tests for IrisFeatures Pydantic model."""
    
    def test_iris_features_valid_creation(self):
        """Test creating valid IrisFeatures instance."""
        features = IrisFeatures(
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2
        )
        assert features.sepal_length == 5.1
        assert features.sepal_width == 3.5
        assert features.petal_length == 1.4
        assert features.petal_width == 0.2
    def test_iris_features_rejects_negative(self):
        """Test IrisFeatures rejects negative values."""
        with pytest.raises(ValidationError):  # Pydantic validation error
            IrisFeatures(
                sepal_length=-5.1,
                sepal_width=3.5,
                petal_length=1.4,
                petal_width=0.2
            )
    
    def test_iris_features_zero_values(self):
        """Test IrisFeatures accepts zero values."""
        features = IrisFeatures(
            sepal_length=0.0,
            sepal_width=0.0,
            petal_length=0.0,
            petal_width=0.0
        )
        assert features.sepal_length == 0.0


class TestBatchIrisFeaturesModel:
    """Tests for BatchIrisFeatures Pydantic model."""
    
    def test_batch_features_valid_creation(self, sample_iris_features):
        """Test creating valid BatchIrisFeatures instance."""
        batch = BatchIrisFeatures(
            samples=[
                IrisFeatures(**sample_iris_features),
                IrisFeatures(**sample_iris_features)
            ]
        )
        assert len(batch.samples) == 2
    
    def test_batch_features_empty_list(self):
        """Test BatchIrisFeatures with empty list."""
        batch = BatchIrisFeatures(samples=[])
        assert len(batch.samples) == 0


class TestModelLoading:
    """Tests for model loading functionality."""
    
    def test_startup_loads_model(self, client):
        """Test that startup event loads the model."""
        # Model should be loaded by fixture
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
    
    def test_predict_without_model_fails(self, sample_iris_features):
        """Test prediction fails when model is not loaded."""
        # Create a client without proper model loading
        import iris_api.app as app_module
        
        # Backup and clear the model
        original_model = app_module.model
        app_module.model = None
        
        with TestClient(app) as test_client:
            response = test_client.post("/predict", json=sample_iris_features)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
        
        # Restore model
        app_module.model = original_model
