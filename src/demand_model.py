import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class DemandPredictionEngine:
    """
    AeroNexus AI — Demand Prediction Engine

    Predicts passenger demand using:
    - Average Fare
    - Distance
    - Market Share
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}

    # ---------------------------------------------------------
    # 1️⃣ Load & Prepare Data
    # ---------------------------------------------------------
    def load_data(self):
        print("Loading data for demand modeling...")

        self.df = pd.read_csv(self.data_path)

        required_cols = [
            "passengers",
            "avg_fare",
            "distance_miles",
            "largest_carrier_market_share",
        ]

        self.df = self.df[required_cols].dropna()

        self.df = self.df[
            (self.df["passengers"] > 0)
            & (self.df["avg_fare"] > 0)
            & (self.df["distance_miles"] > 0)
        ]

        print(f"Clean dataset size: {len(self.df):,}")

    # ---------------------------------------------------------
    # 2️⃣ Train Models
    # ---------------------------------------------------------
    def train_models(self):

        X = self.df[
            ["avg_fare", "distance_miles", "largest_carrier_market_share"]
        ]
        y = self.df["passengers"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        self.models["Linear Regression"] = lr
        self.models["Random Forest"] = rf

        # Evaluate
        for name, model in self.models.items():
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            self.results[name] = {
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4),
            }

        print("Model training complete.")

    # ---------------------------------------------------------
    # 3️⃣ Display Results
    # ---------------------------------------------------------
    def print_results(self):
        print("\n📊 Model Performance Comparison:\n")
        for model_name, metrics in self.results.items():
            print(f"{model_name}")
            print(f"  RMSE: {metrics['RMSE']}")
            print(f"  R²: {metrics['R2']}")
            print("-" * 30)