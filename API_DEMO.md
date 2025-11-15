# FastAPI Model Serving Demo

## Quick Start

1. **Train models first** (if not already done):
   ```bash
   make pipeline_local
   ```

2. **Start the API server**:
   ```bash
   make serve
   ```

3. **Access the interactive documentation**:
   Open http://127.0.0.1:8000/docs in your browser

## API Endpoints

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Make Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "relative_compactness": 0.8,
       "surface_area": 650.0,
       "wall_area": 300.0,
       "roof_area": 150.0,
       "overall_height": 5.0,
       "orientation": 3,
       "glazing_area": 0.25,
       "glazing_area_distribution": 2
     }'
```

### Model Information
```bash
curl http://127.0.0.1:8000/model-info
```

## Python Client Example

```python
import requests

# Test the API
response = requests.post("http://127.0.0.1:8000/predict", json={
    "relative_compactness": 0.8,
    "surface_area": 650.0,
    "wall_area": 300.0,
    "roof_area": 150.0,
    "overall_height": 5.0,
    "orientation": 3,
    "glazing_area": 0.25,
    "glazing_area_distribution": 2
})

print(response.json())
# Expected output: {"heating_load": 15.2, "cooling_load": 18.7, "model_used": "RandomForest"}
```

## Model Artifact Location

The API automatically loads the latest RandomForest model from:
- **MLflow Experiment**: `energy_efficiency_models`
- **Model Type**: `RandomForest` Pipeline
- **Artifact Path**: `mlruns/experiments/<experiment_id>/runs/<run_id>/artifacts/model`

## Input Validation

The API includes input validation:
- **Range validation**: All numeric fields have min/max constraints
- **Type validation**: Ensures correct data types (float/int)
- **Required fields**: All 8 building features are mandatory
- **Error responses**: Returns 422 for validation errors with detailed messages
