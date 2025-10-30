from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from ..dataset import DataLoader
from ..features import DataPreprocessor
from ..plots import VisualEDA
from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_FILE, PROCESSED_DATA_FILE, TARGET_COLS, TEST_SIZE, RANDOM_STATE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelTrainer:
    def __init__(
        self,
        df,
        target_cols=TARGET_COLS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    ):
        self.df = df.copy()
        self.target_cols = target_cols
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = self.X_test = self.Y_train = self.Y_test = None
        self.scaler = StandardScaler()
        self.models = {}

    def list_models(self):
        """
        Print the names of all currently available models for training.
        """
        print("\nCurrent models to train:")
        for name in self.models.keys():
            print(f"- {name}")

    def split_and_scale(self):
        """
        Split the dataset into training and testing sets and apply feature scaling.
        """
        X = self.df.drop(columns=self.target_cols)
        Y = self.df[self.target_cols]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state
        )
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(
            f"Data split and standardized: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test"
        )

    def train_models(self):
        """
        Train Linear Regression, Random Forest, and Gradient Boosting models
        for each target variable (heating_load and cooling_load).
        """
        # Linear Regression
        for target in self.target_cols:
            lr = LinearRegression()
            lr.fit(self.X_train_scaled, self.Y_train[target])
            self.models[f"LinearRegression_{target}"] = lr

        # Random Forest
        for target in self.target_cols:
            rf = RandomForestRegressor(
                n_estimators=600,
                max_depth=12,
                min_samples_split=4,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(self.X_train_scaled, self.Y_train[target])
            self.models[f"RandomForest_{target}"] = rf

        # Gradient Boosting
        for target in self.target_cols:
            gb = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=4,
                random_state=self.random_state,
            )
            gb.fit(self.X_train_scaled, self.Y_train[target])
            self.models[f"GradientBoosting_{target}"] = gb

        print("Models trained successfully.")


class ModelEvaluator:
    def __init__(self, models, X_test_scaled, Y_test):
        self.models = models
        self.X_test_scaled = X_test_scaled
        self.Y_test = Y_test

    def list_models(self):
        """
        Print the names of all trained models available for evaluation.
        """
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


def main(showVisualEDA: bool):
    """
    Execute the complete machine learning pipeline for energy efficiency analysis.
    
    Keyword arguments:
    showVisualEDA -- whether to display visual exploratory data analysis plots (default False)
    """
    data_loader = DataLoader()

    df = data_loader.getDataFrameFromFile(RAW_DATA_DIR / RAW_DATA_FILE)

    print(f"\n\n > Lodaded data - Rows: {df.shape[0]}, Columns: {df.shape[1]}", "\n\n")

    eda = VisualEDA(df)
    data_preprocessor = DataPreprocessor(df)

    data_preprocessor.convert_numeric()

    print(f"\n\n > Data overview before cleansing...", "\n\n")

    eda.overview()

    if showVisualEDA:
        print(f"\n\n > Initializing visual EDA before processing...", "\n\n")

        eda.plot_histograms()
        eda.plot_boxplots()
        eda.plot_correlation_heatmap()

    print(f"\n\n > Initializing data cleansing...", "\n\n")

    data_preprocessor.impute_missing()
    data_preprocessor.detect_outliers()
    data_preprocessor.standardize()

    print(f"\n\n > Data overview after cleansing...", "\n\n")

    eda.overview()

    data_loader.saveDataFrameAsFileWithDVC(
        data_preprocessor.df, PROCESSED_DATA_DIR, PROCESSED_DATA_FILE
    )

    print(f"\n\n > Initializing model training...", "\n\n")

    trainer = ModelTrainer(data_preprocessor.df)
    trainer.split_and_scale()
    trainer.train_models()

    print(f"\n\n > Initializing model evaluation...", "\n\n")

    evaluator = ModelEvaluator(trainer.models, trainer.X_test_scaled, trainer.Y_test)
    evaluator.evaluate_all()

    if showVisualEDA:
        print(f"\n\n > Initializing visual EDA after processing...", "\n\n")

        eda.plot_histograms()
        eda.plot_boxplots()
        eda.plot_correlation_heatmap()

if __name__ == "__main__":
    main(showVisualEDA=True)
