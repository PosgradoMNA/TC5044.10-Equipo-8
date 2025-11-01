from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data file names
RAW_DATA_FILE = "energy_efficiency_modified.csv"
PROCESSED_DATA_FILE = "energy_efficiency_modified.csv"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLS = ["heating_load", "cooling_load"]

# Column naming map
NAMING_MAP = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}

# Numeric columns for preprocessing
NUMERIC_COLS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
    "heating_load",
    "cooling_load",
    "mixed_type_col",
]

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "energy_efficiency_models"
MLFLOW_TRACKING_URI = "./mlruns"
