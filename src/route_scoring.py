import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class RouteScoringEngine:
    """
    AeroNexus AI — Route Intelligence Engine

    Computes:
    - Total Revenue per route
    - Revenue Volatility
    - Average Growth Rate
    - Pricing Power (Fare per Mile)
    - Final Route Expansion Score
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.route_metrics = None

    # ---------------------------------------------------------
    # 1️⃣ Load and Clean Data
    # ---------------------------------------------------------
    def load_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)

        required_cols = [
            "origin_city",
            "destination_city",
            "year",
            "passengers",
            "avg_fare",
            "distance_miles",
        ]

        self.df = self.df[required_cols].dropna()

        self.df = self.df[
            (self.df["passengers"] > 0)
            & (self.df["avg_fare"] > 0)
            & (self.df["distance_miles"] > 0)
        ]

        # Revenue calculation
        self.df["revenue"] = self.df["passengers"] * self.df["avg_fare"]

        print(f"Dataset Loaded: {len(self.df):,} clean records")

    # ---------------------------------------------------------
    # 2️⃣ Compute Core Route Metrics
    # ---------------------------------------------------------
    def compute_route_metrics(self):
        print("Computing route metrics...")

        # Total Revenue
        total_revenue = (
            self.df.groupby(["origin_city", "destination_city"])["revenue"]
            .sum()
            .reset_index()
            .rename(columns={"revenue": "total_revenue"})
        )

        # Revenue Volatility
        volatility = (
            self.df.groupby(["origin_city", "destination_city"])["revenue"]
            .std()
            .reset_index()
            .rename(columns={"revenue": "revenue_volatility"})
        )

        # Yearly Revenue for Growth
        yearly = (
            self.df.groupby(
                ["origin_city", "destination_city", "year"]
            )["revenue"]
            .sum()
            .reset_index()
        )

        yearly = yearly.sort_values(
            by=["origin_city", "destination_city", "year"]
        )

        yearly["growth_rate"] = (
            yearly.groupby(["origin_city", "destination_city"])["revenue"]
            .pct_change()
        )

        growth = (
            yearly.groupby(["origin_city", "destination_city"])[
                "growth_rate"
            ]
            .mean()
            .reset_index()
            .rename(columns={"growth_rate": "avg_growth"})
        )

        # Pricing Power
        self.df["fare_per_mile"] = (
            self.df["avg_fare"] / self.df["distance_miles"]
        )

        pricing = (
            self.df.groupby(["origin_city", "destination_city"])[
                "fare_per_mile"
            ]
            .mean()
            .reset_index()
            .rename(columns={"fare_per_mile": "avg_fare_per_mile"})
        )

        # Merge All Metrics
        self.route_metrics = (
            total_revenue.merge(
                volatility, on=["origin_city", "destination_city"]
            )
            .merge(growth, on=["origin_city", "destination_city"])
            .merge(pricing, on=["origin_city", "destination_city"])
        )

        self.route_metrics = self.route_metrics.dropna()

        print(
            f"Computed metrics for {len(self.route_metrics):,} routes"
        )

    # ---------------------------------------------------------
    # 3️⃣ Generate Route Expansion Score
    # ---------------------------------------------------------
    def generate_route_score(self):
        print("Generating route intelligence score...")

        scaler = MinMaxScaler()

        metrics = self.route_metrics[
            [
                "total_revenue",
                "revenue_volatility",
                "avg_growth",
                "avg_fare_per_mile",
            ]
        ]

        scaled = scaler.fit_transform(metrics)

        scaled_df = pd.DataFrame(
            scaled,
            columns=[
                "rev_norm",
                "vol_norm",
                "growth_norm",
                "price_norm",
            ],
        )

        # Stability = inverse of volatility
        scaled_df["stability_norm"] = 1 - scaled_df["vol_norm"]

        # Weighted Route Score
        self.route_metrics["route_score"] = (
            0.35 * scaled_df["rev_norm"]
            + 0.25 * scaled_df["growth_norm"]
            + 0.20 * scaled_df["stability_norm"]
            + 0.20 * scaled_df["price_norm"]
        )

        print("Route scoring complete.")

    # ---------------------------------------------------------
    # 4️⃣ Get Ranked Routes
    # ---------------------------------------------------------
    def get_top_routes(self, top_n: int = 10):
        ranked = self.route_metrics.sort_values(
            by="route_score", ascending=False
        )

        return ranked.head(top_n)