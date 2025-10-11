# TC5044.10-Equipo-8
Energy Efficiency Analysis - Machine Learning Pipeline

## Overview
This project implements a complete machine learning pipeline for analyzing energy efficiency in buildings. It includes data preprocessing, exploratory data analysis, model training, and evaluation using Linear Regression and Random Forest algorithms.

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
3. Place it in the `src/data/` folder

### 4. Run the Pipeline
```bash
cd src
python main.py
```

## Project Structure
```
src/
├── main.py                    # Main execution pipeline
├── data/                      # Data directory
│   └── energy_efficiency_modified.csv
├── handlers/
│   ├── data_loader.py         # Data loading and saving
│   ├── data_preprocessor.py   # Data cleaning and preprocessing
│   ├── model_trainer.py       # Model training
│   ├── model_evaluator.py     # Model evaluation
│   └── visual_eda.py          # Exploratory data analysis
```

## Features
- **Data Loading**: CSV loading with column renaming
- **Data Preprocessing**: Missing value imputation, outlier detection, standardization
- **Visual EDA**: Histograms, boxplots, correlation heatmaps
- **Model Training**: Linear Regression and Random Forest
- **Model Evaluation**: R², RMSE, MAE metrics
- **Version Control**: DVC integration for data versioning

## Models
- **Linear Regression**: For heating and cooling load prediction
- **Random Forest**: Ensemble method with 600 estimators

## Target Variables
- `heating_load`: Building heating energy requirements
- `cooling_load`: Building cooling energy requirements
