import pandas as pd
import numpy as np
import energy_efficiency
from energy_efficiency.features import DataPreprocessor


class TestDataPreprocessor:
    def test_convert_numeric(self):
        """Test numeric conversion functionality."""
        test_data = pd.DataFrame(
            {
                "relative_compactness": [0.8, 0.9, "0.7"],
                "surface_area": [600, "700", 800],
                "mixed_type_col": ["1.5", "2.0", "invalid"],
            }
        )

        original_numeric_cols = energy_efficiency.config.NUMERIC_COLS
        energy_efficiency.config.NUMERIC_COLS = [
            "relative_compactness",
            "surface_area",
            "mixed_type_col",
        ]

        try:
            preprocessor = DataPreprocessor(test_data)
            preprocessor.convert_numeric()

            assert preprocessor.df["relative_compactness"].dtype == float
            assert preprocessor.df["surface_area"].dtype == float
            assert pd.isna(
                preprocessor.df["mixed_type_col"].iloc[2]
            )
        finally:
            energy_efficiency.config.NUMERIC_COLS = original_numeric_cols

    def test_impute_missing(self):
        """Test missing value imputation."""
        test_data = pd.DataFrame(
            {
                "relative_compactness": [0.8, np.nan, 0.7],
                "heating_load": [20, 25, np.nan],
            }
        )

        original_numeric_cols = energy_efficiency.config.NUMERIC_COLS
        energy_efficiency.config.NUMERIC_COLS = ["relative_compactness", "heating_load"]

        try:
            preprocessor = DataPreprocessor(test_data)
            missing_before = preprocessor.df.isna().sum().sum()

            preprocessor.impute_missing()
            missing_after = preprocessor.df.isna().sum().sum()

            assert missing_after < missing_before
            assert missing_after == 0
        finally:
            energy_efficiency.config.NUMERIC_COLS = original_numeric_cols

    def test_detect_outliers(self):
        """Test outlier detection and removal."""
        test_data = pd.DataFrame(
            {
                "relative_compactness": [0.8, 0.9, 0.7, 0.85],
                "heating_load": [20, 25, 1000, 22],
            }
        )

        original_numeric_cols = energy_efficiency.config.NUMERIC_COLS
        energy_efficiency.config.NUMERIC_COLS = ["relative_compactness", "heating_load"]

        try:
            preprocessor = DataPreprocessor(test_data)
            initial_rows = len(preprocessor.df)

            preprocessor.detect_outliers()

            assert len(preprocessor.df) < initial_rows
            assert len(preprocessor.outliers) > 0
        finally:
            energy_efficiency.config.NUMERIC_COLS = original_numeric_cols
