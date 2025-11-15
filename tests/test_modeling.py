import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from energy_efficiency.modeling.train import ModelTrainer
from energy_efficiency.modeling.predict import ModelEvaluator


class TestModelTrainer:
    def test_initialization(self, sample_data):
        """Test ModelTrainer initialization."""
        sample_data = sample_data.rename(
            columns={
                "X1": "relative_compactness",
                "X2": "surface_area",
                "Y1": "heating_load",
                "Y2": "cooling_load",
            }
        )

        trainer = ModelTrainer(sample_data)

        assert trainer.target_cols == ["heating_load", "cooling_load"]
        assert trainer.test_size == 0.2
        assert len(trainer.feature_cols) == len(sample_data.columns) - 2

    def test_split_data(self, sample_data):
        """Test data splitting functionality."""
        sample_data = sample_data.rename(
            columns={
                "X1": "relative_compactness",
                "Y1": "heating_load",
                "Y2": "cooling_load",
            }
        )

        trainer = ModelTrainer(sample_data)
        trainer.split_data()

        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert len(trainer.X_train) == 80  # 80% of 100
        assert len(trainer.X_test) == 20  # 20% of 100

    def test_build_preprocessor(self, sample_data):
        """Test preprocessor pipeline creation."""
        sample_data = sample_data.rename(
            columns={"Y1": "heating_load", "Y2": "cooling_load"}
        )

        trainer = ModelTrainer(sample_data)
        preprocessor = trainer._build_preprocessor()

        assert hasattr(preprocessor, "fit")
        assert hasattr(preprocessor, "transform")


class TestModelEvaluator:
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        models = {"test_model": None}
        X_test = pd.DataFrame({"feature": [1, 2, 3]})
        Y_test = pd.DataFrame({"target": [1, 2, 3]})

        evaluator = ModelEvaluator(models, X_test, Y_test)

        assert evaluator.models == models
        assert len(evaluator.X_test) == 3

    def test_evaluate_model(self):
        """Test single model evaluation metrics."""
        models = {}
        X_test = pd.DataFrame({"feature": [1, 2, 3]})
        Y_test = pd.DataFrame({"target": [1, 2, 3]})

        evaluator = ModelEvaluator(models, X_test, Y_test)

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])

        r2, rmse, mae = evaluator.evaluate_model(y_true, y_pred, "test", "target")

        assert 0 <= r2 <= 1
        assert rmse > 0
        assert mae > 0
