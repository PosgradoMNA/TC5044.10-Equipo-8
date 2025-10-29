"""
Autor: Ricardo Aguilar
Rol: Data Scientist
Archivo: run_experiments_ricardo.py
Descripción:
Script personal para registrar experimentos con MLflow y validar reproducibilidad (Fase 2).
No modifica el código del equipo, solo genera evidencias del seguimiento de experimentos.
"""

from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split
from src.handlers.data_loader import DataLoader
from src.handlers.data_preprocessor import DataPreprocessor
from src.handlers.model_trainer import ModelTrainer
from src.handlers.model_evaluator import ModelEvaluator

def main():
    # === 1. Configuración inicial ===
    data_path = Path("src/data/energy_efficiency_modified.csv")
    mlflow.set_experiment("Energy Efficiency – Ricardo Aguilar")

    # === 2. Carga de datos ===
    loader = DataLoader(str(data_path))
    df = loader.load_data()
    X, y = loader.split_features_target(df, target_col="heating_load")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 3. Preprocesamiento ===
    pre = DataPreprocessor()
    X_train_p = pre.preprocess(X_train)
    X_test_p = pre.transform(X_test) if hasattr(pre, "transform") else pre.preprocess(X_test)

    evaluator = ModelEvaluator()

    # === 4. Experimento 1: Regresión Lineal ===
    with mlflow.start_run(run_name="LinearRegression_Baseline"):
        trainer = ModelTrainer(X_train_p, y_train)
        lr_model = trainer.train_linear()
        preds_lr = lr_model.predict(X_test_p)
        evaluator.evaluate(y_test, preds_lr, "LinearRegression")
        mlflow.log_param("algorithm", "LinearRegression")

    # === 5. Experimento 2: Random Forest ===
    with mlflow.start_run(run_name="RandomForest_600"):
        trainer = ModelTrainer(X_train_p, y_train)
        rf_model = trainer.train_random_forest(n_estimators=600, random_state=42)
        preds_rf = rf_model.predict(X_test_p)
        evaluator.evaluate(y_test, preds_rf, "RandomForest_600")
        mlflow.log_param("algorithm", "RandomForest")
        mlflow.log_param("n_estimators", 600)

    print("\n✅ Experimentos completados y registrados en MLflow con éxito.")

if __name__ == "__main__":
    main()
