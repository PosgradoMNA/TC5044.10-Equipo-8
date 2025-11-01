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
```

### 3. Download Dataset
1. Download the Energy Efficiency dataset from: https://experiencia21.tec.mx/courses/624720/pages/datasets-por-equipos (team 8)
2. Extract the CSV file and rename it to `energy_efficiency_modified.csv`
3. Place it in the `data/raw/` folder

### 4. Run the Pipeline
```bash
make pipeline
```

Or directly:
```bash
python -m energy_efficiency.modeling.train
```

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
