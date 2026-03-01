import pandas as pd
import numpy as np
import itertools
import math
import hashlib


class RouteExpansionSimulator:
    """
    AeroNexus AI — Distance-Aware Route Expansion Simulator
    """

    def __init__(self, graph, demand_model, route_df):
        self.graph = graph
        self.model = demand_model
        self.route_df = route_df
        self.demand_df = demand_model.df

        # Generate deterministic airport coordinates
        self.airport_coords = self._generate_airport_coordinates()

    # -----------------------------------------------------
    # Generate Deterministic Coordinates from Airport Name
    # -----------------------------------------------------
    def _generate_airport_coordinates(self):

        coords = {}

        for airport in self.graph.nodes():

            hash_val = int(hashlib.md5(airport.encode()).hexdigest(), 16)

            # Approximate US bounds
            lat = 25 + (hash_val % 2400) / 100   # 25 to 49
            lon = -124 + (hash_val % 5800) / 100  # -124 to -66

            coords[airport] = (lat, lon)

        return coords

    # -----------------------------------------------------
    # Haversine Distance Formula
    # -----------------------------------------------------
    def _haversine(self, lat1, lon1, lat2, lon2):

        R = 3958.8  # Earth radius in miles

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)

        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1)
            * math.cos(phi2)
            * math.sin(delta_lambda / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # -----------------------------------------------------
    # Estimate Fare Based on Distance
    # -----------------------------------------------------
    def _estimate_fare(self, distance):

        avg_fare_per_mile = (
            self.route_df["avg_fare"] /
            self.route_df["distance_miles"]
        ).mean()

        return distance * avg_fare_per_mile

    # -----------------------------------------------------
    # Simulate Route Expansion
    # -----------------------------------------------------
    def simulate_new_routes(self, top_n=10):

        degree_dict = dict(self.graph.degree())

        # Focus on top 30 busiest airports
        sorted_airports = sorted(
            degree_dict,
            key=degree_dict.get,
            reverse=True
        )

        airports = sorted_airports[:30]

        potential_routes = []

        rf_model = self.model.models["Random Forest"]

        avg_market_share = self.demand_df[
            "largest_carrier_market_share"
        ].mean()

        for a, b in itertools.combinations(airports, 2):

            if self.graph.has_edge(a, b):
                continue

            lat1, lon1 = self.airport_coords[a]
            lat2, lon2 = self.airport_coords[b]

            distance = self._haversine(lat1, lon1, lat2, lon2)

            fare = self._estimate_fare(distance)

            features = pd.DataFrame([{
                "avg_fare": fare,
                "distance_miles": distance,
                "largest_carrier_market_share": avg_market_share,
            }])

            predicted_passengers = rf_model.predict(features)[0]

            estimated_revenue = predicted_passengers * fare

            potential_routes.append({
                "origin": a,
                "destination": b,
                "distance_miles": round(distance, 2),
                "estimated_fare": round(fare, 2),
                "predicted_passengers": round(predicted_passengers, 0),
                "estimated_revenue": round(estimated_revenue, 2),
            })

        results_df = pd.DataFrame(potential_routes)

        if results_df.empty:
            return pd.DataFrame()

        results_df = results_df.sort_values(
            by="estimated_revenue",
            ascending=False
        )

        return results_df.head(top_n)