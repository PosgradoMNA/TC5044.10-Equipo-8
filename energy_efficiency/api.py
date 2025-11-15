from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from .config import MODELS_DIR, MLFLOW_TRACKING_URI
import mlflow

app = FastAPI(
    title="Energy Efficiency Prediction API",
    description="API for predicting building heating and cooling loads",
    version="1.0.0",
)


class BuildingFeatures(BaseModel):
    relative_compactness: float = Field(
        ..., ge=0.6, le=1.0, description="Building relative compactness"
    )
    surface_area: float = Field(..., ge=500, le=900, description="Surface area in m²")
    wall_area: float = Field(..., ge=200, le=400, description="Wall area in m²")
    roof_area: float = Field(..., ge=100, le=250, description="Roof area in m²")
    overall_height: float = Field(
        ..., ge=3, le=7, description="Overall height in meters"
    )
    orientation: int = Field(..., ge=2, le=5, description="Building orientation")
    glazing_area: float = Field(..., ge=0, le=0.4, description="Glazing area ratio")
    glazing_area_distribution: int = Field(
        ..., ge=0, le=5, description="Glazing area distribution"
    )


class PredictionResponse(BaseModel):
    heating_load: float = Field(..., description="Predicted heating load")
    cooling_load: float = Field(..., description="Predicted cooling load")
    model_used: str = Field(..., description="Model used for prediction")


# Load all models on startup
models = {}
available_models = ["LinearRegression", "RandomForest", "GradientBoosting"]


@app.on_event("startup")
async def load_models():
    global models
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        experiment = mlflow.get_experiment_by_name("energy_efficiency_models")
        if experiment:
            for model_name in available_models:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{model_name}'",
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]
                    model_uri = f"runs:/{run_id}/model"
                    models[model_name] = mlflow.sklearn.load_model(model_uri)
                    print(f"Loaded {model_name} from run: {run_id}")

        print(f"Loaded models: {list(models.keys())}")
    except Exception as e:
        print(f"Error loading models: {e}")


@app.get("/")
async def root():
    return {"message": "Energy Efficiency Prediction API", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if models else "unhealthy",
        "models_loaded": list(models.keys()),
        "available_models": available_models,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    features: BuildingFeatures,
    model_name: str = Query(
        "RandomForest",
        description="Model to use for prediction (default: RandomForest)",
        enum=available_models,
    ),
):
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not available. Available models: {list(models.keys())}",
        )

    try:
        input_data = pd.DataFrame([features.dict()])
        input_data["mixed_type_col"] = 0.0

        prediction = models[model_name].predict(input_data)

        return PredictionResponse(
            heating_load=float(prediction[0][0]),
            cooling_load=float(prediction[0][1]),
            model_used=model_name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
async def model_info():
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    return {
        "available_models": list(models.keys()),
        "model_types": {name: type(model).__name__ for name, model in models.items()},
        "features": list(BuildingFeatures.__fields__.keys()),
        "targets": ["heating_load", "cooling_load"],
    }
