import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import NUMERIC_COLS


class DataPreprocessor:
    NUMERIC_COLS = NUMERIC_COLS

    def __init__(self, df):
        self.df = df
        self.outliers = []

    def convert_numeric(self):
        """
        Convert specified columns to numeric float type.
        
        Converts all columns in NUMERIC_COLS to float, handling errors by coercing to NaN.
        """
        print(f"\nConverting numeric values...", "\n")
        for col in self.NUMERIC_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(float)

    def impute_missing(self):
        """
        Fill missing values in numeric columns with their median values.
        
        Prints the number of missing values before and after imputation.
        """
        print(f"\nInitializing imputation of values...", "\n")

        missing_before = self.df.isna().sum().sum()
        print(f"\nMissing values before imputation: {missing_before}", "\n")

        for col in self.NUMERIC_COLS:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        missing_after = self.df.isna().sum().sum()

        print(f"\nMissing values after imputation: {missing_after}", "\n")

    def detect_outliers(self):
        """
        Detect and remove outliers using the Interquartile Range (IQR) method.
        
        Identifies outliers as values beyond 1.5 * IQR from Q1 and Q3.
        Removes entire rows containing outliers and stores their indices.
        """
        print(f"\nInitializing outlier analysis...", "\n")
        outlier_rows = set()
        for col in self.NUMERIC_COLS:
            data = self.df[col].values
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = np.where((data < lower) | (data > upper))[0]
            outlier_rows.update(outliers)
        self.outliers = list(outlier_rows)
        print(f"\nRows detected as outliers: {len(self.outliers)}", "\n")
        self.df.drop(index=self.outliers, inplace=True)
        print(f"\nOutliers removed", "\n")

    def standardize(self):
        """
        Standardize numeric columns using StandardScaler.
        
        Applies z-score normalization to all numeric columns to have mean=0 and std=1.
        """
        print(
            f"\nStandardized numeric columns using StandardScaler for the following columns: {self.NUMERIC_COLS}",
            "\n",
        )
        scaler = StandardScaler()
        self.df[self.NUMERIC_COLS] = scaler.fit_transform(self.df[self.NUMERIC_COLS])
