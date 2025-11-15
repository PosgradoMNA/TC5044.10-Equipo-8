import pandas as pd
import numpy as np
import mlflow
import energy_efficiency
from energy_efficiency.dataset import DataLoader
from energy_efficiency.features import DataPreprocessor
from energy_efficiency.modeling.train import ModelTrainer
from energy_efficiency.modeling.predict import ModelEvaluator


class TestEndToEndPipeline:
    """Integration tests for the complete ML pipeline."""

    def test_complete_pipeline_flow(self, sample_csv_file, temp_dir):
        """Test end-to-end pipeline execution."""
        original_start_run = mlflow.start_run
        original_log_param = mlflow.log_param
        original_log_metric = mlflow.log_metric
        original_sklearn_log_model = mlflow.sklearn.log_model

        mlflow.start_run = lambda *args, **kwargs: type(
            "MockRun", (), {"__enter__": lambda x: x, "__exit__": lambda x, *args: None}
        )()
        mlflow.log_param = lambda *args, **kwargs: None
        mlflow.log_metric = lambda *args, **kwargs: None
        mlflow.sklearn.log_model = lambda *args, **kwargs: None

        original_numeric_cols = energy_efficiency.config.NUMERIC_COLS
        energy_efficiency.config.NUMERIC_COLS = [
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
        ]

        try:
            loader = DataLoader()
            df = loader.getDataFrameFromFile(sample_csv_file)
            assert len(df) > 0

            preprocessor = DataPreprocessor(df)
            preprocessor.convert_numeric()
            preprocessor.impute_missing()
            preprocessor.detect_outliers()

            processed_df = preprocessor.df
            assert len(processed_df) <= len(df)

            trainer = ModelTrainer(processed_df)
            trainer.split_data()
            trainer.train_models()

            assert len(trainer.models) > 0
            assert "LinearRegression" in trainer.models

            evaluator = ModelEvaluator(
                trainer.models,
                trainer.X_test,
                trainer.Y_test,
                trainer.validation_reports,
            )
            results = evaluator.evaluate_all()

            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            assert "R2" in results.columns

        finally:
            mlflow.start_run = original_start_run
            mlflow.log_param = original_log_param
            mlflow.log_metric = original_log_metric
            mlflow.sklearn.log_model = original_sklearn_log_model
            energy_efficiency.config.NUMERIC_COLS = original_numeric_cols

    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal valid dataset."""
        np.random.seed(42)
        minimal_data = pd.DataFrame(
            {
                "relative_compactness": np.random.uniform(0.6, 1.0, 20),
                "surface_area": np.random.uniform(500, 800, 20),
                "heating_load": np.random.uniform(10, 40, 20),
                "cooling_load": np.random.uniform(10, 45, 20),
            }
        )

        mlflow.start_run = lambda *args, **kwargs: type(
            "MockRun", (), {"__enter__": lambda x: x, "__exit__": lambda x, *args: None}
        )()
        mlflow.log_param = lambda *args, **kwargs: None
        mlflow.log_metric = lambda *args, **kwargs: None
        mlflow.sklearn.log_model = lambda *args, **kwargs: None

        original_numeric_cols = energy_efficiency.config.NUMERIC_COLS
        energy_efficiency.config.NUMERIC_COLS = [
            "relative_compactness",
            "surface_area",
            "heating_load",
            "cooling_load",
        ]

        try:
            # Process through pipeline
            preprocessor = DataPreprocessor(minimal_data)
            preprocessor.convert_numeric()

            trainer = ModelTrainer(preprocessor.df)
            trainer.split_data()

            assert trainer.X_train is not None
            assert len(trainer.X_train) > 0
        finally:
            energy_efficiency.config.NUMERIC_COLS = original_numeric_cols

    def test_data_consistency_through_pipeline(self, sample_csv_file):
        """Test data consistency maintained through pipeline stages."""
        original_numeric_cols = energy_efficiency.config.NUMERIC_COLS
        energy_efficiency.config.NUMERIC_COLS = [
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
        ]

        try:
            loader = DataLoader()
            original_df = loader.getDataFrameFromFile(sample_csv_file)

            preprocessor = DataPreprocessor(original_df)
            preprocessor.convert_numeric()

            expected_cols = ["relative_compactness", "heating_load", "cooling_load"]
            for col in expected_cols:
                if col in original_df.columns:
                    assert col in preprocessor.df.columns
        finally:
            energy_efficiency.config.NUMERIC_COLS = original_numeric_cols
