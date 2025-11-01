# TC5044.10-Equipo-8
Energy Efficiency Analysis - Machine Learning Pipeline

## Overview
This project implements a complete machine learning pipeline for analyzing energy efficiency in buildings. It now leverages scikit-learn Pipelines to automate preprocessing, model training, and evaluation with Linear Regression, Random Forest, and Gradient Boosting regressors.

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install "dvc[s3]"
```

### 3. Configure AWS (Team Members Only)
```bash
# Configure your team profile (replace X with your team number)
aws configure --profile equipoX
# Enter your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, region: us-east-2, format: json
```

### 4. Pull Data from S3
```bash
# Pull latest datasets from team S3 bucket
python -m dvc pull
```

### 5. Run the Pipeline
```bash
make pipeline
```

### 6. Push Changes to S3 (After Processing)
```bash
# Automatically saves processed data and pushes to S3
python -m dvc push
git add data/processed/energy_efficiency_modified.csv.dvc
git commit -m "Update processed data"
```

## Automated Workflow

### Full Pipeline (Recommended)
```bash
make pipeline
```
This command automatically:
1. Pulls latest data from S3 (`dvc pull`)
2. Runs the complete ML pipeline
3. Pushes processed results back to S3 (`dvc push`)

### Individual Commands
```bash
make sync_data_from_s3    # Pull data from S3
make sync_data_to_s3      # Push data to S3  
make pipeline_local       # Run pipeline without S3 sync
```

### Team Collaboration Workflow
1. **Start work**: `make sync_data_from_s3` (get latest data)
2. **Run pipeline**: `make pipeline_local` (process locally)
3. **Share results**: `make sync_data_to_s3` (push to S3)
4. **Commit changes**: `git add *.dvc && git commit -m "Update data"`

## Project Structure
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         energy_efficiency and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── energy_efficiency   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes energy_efficiency a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Features
- **Data Loading**: CSV loading with consistent column renaming
- **Data Preprocessing**: Type coercion, robust outlier handling, and automatic imputation/scaling encoded inside scikit-learn Pipelines
- **Visual EDA**: Histograms, boxplots, correlation heatmaps
- **Model Training**: Linear Regression, Random Forest, and Gradient Boosting, all trained through reusable Pipelines
- **Model Evaluation**: Hold-out metrics (R², RMSE, MAE) plus 5-fold cross-validation summaries
- **Version Control**: DVC integration for data versioning

## Models
- **Linear Regression**: Baseline multi-output regressor
- **Random Forest**: Ensemble method with 600 estimators
- **Gradient Boosting**: Gradient-boosted trees with tuned learning rate and depth

## Target Variables
- `heating_load`: Building heating energy requirements
- `cooling_load`: Building cooling energy requirements

## MLflow Experiment Tracking

### View Experiments
```bash
make pipeline             # Run full pipeline + auto-start MLflow UI
make mlflow_ui            # Start MLflow UI only (view existing experiments)
# Open http://127.0.0.1:5000 in your browser
```

### What's Tracked
- **Parameters**: test_size, random_state, n_features, n_samples
- **Metrics**: Cross-validation R², RMSE, MAE for each model
- **Models**: Serialized scikit-learn pipelines
- **Artifacts**: Model files for deployment

### Compare Models
The MLflow UI shows:
- Model performance comparison
- Parameter impact analysis  
- Model versioning and deployment

### Access MLflow UI
1. Run the pipeline: `make pipeline`
2. MLflow UI will automatically open at: **http://localhost:5000**
3. Browse experiments and compare model performance
4. Press `Ctrl+C` in terminal to stop the UI server
