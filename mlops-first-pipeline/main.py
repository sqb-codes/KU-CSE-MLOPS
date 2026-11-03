from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np
import os
from contextlib import asynccontextmanager
from typing import Callable


# MLFlow Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:7001"
)

# model:/iris-classifier@champion
MODEL_URI = "models:/iris-classifier@champion"

def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    return model

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

def create_app(
        model_loader: Callable=load_mlflow_model,
        version_loader: Callable=None):

    if version_loader is None:
        def version_loader():
            client = mlflow.MlflowClient(
                        tracking_uri=MLFLOW_TRACKING_URI
                    )
            champion_version = client.get_model_version_by_alias(
                name="iris-classifier",
                alias="champion"
            )
            return champion_version.version

    @asynccontextmanager
    def lifespan(app:FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load model: {ex}"
            )

        app.state.model_version = version_loader()
        yield

    app = FastAPI(
        title="Iris Classification API",
        description="Iris prediction API powered by MLFlow",
        version="3.0.0",
        lifespan=lifespan
    )
        
    @app.get("/")
    def ready_to_serve():
        return {"status": "Ready to serve"}

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_name": "iris-classification",
            "model_version": app.state.model_version
            }

    # Endpoint for making predictions with the Iris Random Forest model
    @app.post("/predict")
    def predict(data: IrisInput, request: Request):
        model = request.app.state.model
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]
        prediction = int(model.predict(features)[0])
        print("Prediction is",prediction)
        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction]
        }


# Production application instance
app = create_app()