import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.models.model import CreditRiskModel

client = TestClient(app)

class TestAPI:
    """Test cases for the Credit Risk API"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_api_info(self):
        """Test API info endpoint"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_predict_valid_data(self):
        """Test prediction with valid data"""
        valid_data = {
            "annual_inc": 75000,
            "int_rate": 12.5,
            "credit_history_length": 5.5,
            "purpose": "debt_consolidation",
            "verification_status": "verified"
        }
        
        response = client.post("/api/v1/predict", json=valid_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "credit_score" in data
        assert "default_probability" in data
        assert "risk_level" in data
        assert isinstance(data["credit_score"], (int, float))
        assert 0 <= data["default_probability"] <= 1
    
    def test_predict_missing_required_field(self):
        """Test prediction with missing required field"""
        invalid_data = {
            "annual_inc": 75000,
            "int_rate": 12.5,
            # Missing credit_history_length
            "purpose": "debt_consolidation",
            "verification_status": "verified"
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_data_types(self):
        """Test prediction with invalid data types"""
        invalid_data = {
            "annual_inc": "not_a_number",
            "int_rate": 12.5,
            "credit_history_length": 5.5,
            "purpose": "debt_consolidation",
            "verification_status": "verified"
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_out_of_range_values(self):
        """Test prediction with out of range values"""
        invalid_data = {
            "annual_inc": -1000,  # Negative income
            "int_rate": 12.5,
            "credit_history_length": 5.5,
            "purpose": "debt_consolidation",
            "verification_status": "verified"
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict(self):
        """Test batch prediction"""
        batch_data = {
            "applications": [
                {
                    "annual_inc": 75000,
                    "int_rate": 12.5,
                    "credit_history_length": 5.5,
                    "purpose": "debt_consolidation",
                    "verification_status": "verified"
                },
                {
                    "annual_inc": 50000,
                    "int_rate": 15.0,
                    "credit_history_length": 3.0,
                    "purpose": "car",
                    "verification_status": "not_verified"
                }
            ]
        }
        
        response = client.post("/api/v1/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_applications" in data
        assert len(data["predictions"]) == 2
    
    def test_batch_predict_too_large(self):
        """Test batch prediction with too many applications"""
        applications = []
        for i in range(101):  # Exceeds limit of 100
            applications.append({
                "annual_inc": 75000,
                "int_rate": 12.5,
                "credit_history_length": 5.5,
                "purpose": "debt_consolidation",
                "verification_status": "verified"
            })
        
        batch_data = {"applications": applications}
        
        response = client.post("/api/v1/predict/batch", json=batch_data)
        assert response.status_code == 400
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "features_used" in data
        assert "scoring_params" in data
    
    def test_feature_importance(self):
        """Test feature importance endpoint"""
        response = client.get("/api/v1/model/features/importance")
        assert response.status_code == 200
        
        data = response.json()
        assert "feature_importance" in data
        assert "total_features" in data
        assert isinstance(data["feature_importance"], dict)
    
    def test_available_features(self):
        """Test available features endpoint"""
        response = client.get("/api/v1/model/features")
        assert response.status_code == 200
        
        data = response.json()
        assert "features" in data
        assert "required_features" in data
        assert "optional_features" in data

class TestModel:
    """Test cases for the Credit Risk Model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = CreditRiskModel()
        assert model is not None
        assert model.scoring_params is not None
        assert len(model.final_woe_features) > 0
    
    def test_model_prediction(self):
        """Test model prediction"""
        model = CreditRiskModel()
        
        test_data = {
            "annual_inc": 75000,
            "int_rate": 12.5,
            "credit_history_length": 5.5,
            "purpose": "debt_consolidation",
            "verification_status": "verified"
        }
        
        result = model.predict_score(test_data)
        
        assert "credit_score" in result
        assert "default_probability" in result
        assert "risk_level" in result
        assert isinstance(result["credit_score"], (int, float))
        assert 0 <= result["default_probability"] <= 1
    
    def test_model_validation(self):
        """Test model input validation"""
        model = CreditRiskModel()
        
        # Valid data
        valid_data = {
            "annual_inc": 75000,
            "int_rate": 12.5,
            "purpose": "debt_consolidation"
        }
        
        is_valid, errors = model.validate_input(valid_data)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid data
        invalid_data = {
            "annual_inc": -1000,  # Negative income
            "int_rate": 12.5,
            "purpose": "debt_consolidation"
        }
        
        is_valid, errors = model.validate_input(invalid_data)
        assert not is_valid
        assert len(errors) > 0
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        model = CreditRiskModel()
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # All importance values should be non-negative
        for feature, imp in importance.items():
            assert imp >= 0
