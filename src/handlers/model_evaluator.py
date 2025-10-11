import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    def __init__(self, models, X_test_scaled, Y_test):
        self.models = models
        self.X_test_scaled = X_test_scaled
        self.Y_test = Y_test

    def list_models(self):
        """
        Print the names of all trained models available for evaluation."""
        print("\nTrained models:")
        for name in self.models.keys():
            print(f"- {name}")

    def evaluate_model(self, y_true, y_pred, name):
        """
        Calculate and display regression metrics for a single model.

        Keyword arguments:
        y_true -- actual target values
        y_pred -- predicted target values
        name -- name of the model being evaluated

        Returns tuple of (R², RMSE, MAE) metrics.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{name:<30} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        return r2, rmse, mae

    def evaluate_all(self):
        """
        Evaluate all trained models and return results as a DataFrame.

        Calculates R², RMSE, and MAE for each model and returns
        a pandas DataFrame with all evaluation metrics.
        """
        self.list_models()
        print("\nModel evaluation:\n")
        results = []
        for name, model in self.models.items():
            target = "_".join(name.split("_")[1:])
            y_true = self.Y_test[target]
            y_pred = model.predict(self.X_test_scaled)
            r2, rmse, mae = self.evaluate_model(y_true, y_pred, name)
            results.append([name, r2, rmse, mae])
        return pd.DataFrame(results, columns=["Model", "R2", "RMSE", "MAE"])
