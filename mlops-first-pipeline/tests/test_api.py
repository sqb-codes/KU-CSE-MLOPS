# Import TestClient so we can send simulated HTTP requests directly to FastAPI
from fastapi.testclient import TestClient

import pytest

# Import the application object defined in main.py
from main import create_app

# Create a reusable test client for all test functions.
# client = TestClient(app)

# The tests will no longer require a running MLFlow server
# They test API independently using a fake model
class FakeIrisModel:
    def predict(self, features):
        return [0]

def fake_model_loader():
    return FakeIrisModel()

def fake_version_loader():
    return "test-version"

@pytest.fixture
def client():
    app = create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint():
    """
    Verify that the application is running and can respond to a health-check request
    """
    response = client.get("/health")
    # A successful HTTP request should return status code 200
    assert response.status_code == 200
    
    # Now check response body
    assert response.json()["status"] == {"status": "healthy"}
    assert response.json()["model_version"] == "test_version"


def test_valid_iris_prediction():
    """
    Verify that valid flower measurements produce a prediction
    """
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }
    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Convert the JSON response into a Python Dictionary
    result = response.json()

    # Check the response contract: both fields must exists
    assert "prediction" in result
    assert "class_name" in result

    # The predicted class must be one of the three Iris classes
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    # The predicted integere must be between 0 and 2
    assert result["prediction"] in [0,1,2]


def test_invalid_iris_input():
    """
    Verify that invalid inputs are rejected by FastAPI
    """
    payload = {
            "sepal_length": -2.3,
            "sepal_width": 1.4,
            "petal_length": 3.4,
            "petal_width": 0.4
        }
    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)
    # FastAPI normally returns HTTP 422 for validation error
    assert response.status_code == 422