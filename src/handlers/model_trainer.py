from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class ModelTrainer:
    def __init__(
        self,
        df,
        target_cols=["heating_load", "cooling_load"],
        test_size=0.2,
        random_state=42,
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
        Print the names of all currently available models for training."""
        print("\nCurrent models to train:")
        for name in self.models.keys():
            print(f"- {name}")

    def split_and_scale(self):
        """
        Split the dataset into training and testing sets and apply feature scaling.
        
        Separates features from target variables, performs train-test split,
        and applies StandardScaler to normalize feature values.
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
        Train Linear Regression and Random Forest models for each target variable.
        
        Creates separate models for heating_load and cooling_load predictions
        using both Linear Regression and Random Forest algorithms.
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

        print("Models trained successfully.")
