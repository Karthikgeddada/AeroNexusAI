import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.route_scoring import RouteScoringEngine
from src.demand_model import DemandPredictionEngine
from src.network_graph import AirlineNetworkGraph
from src.optimization_engine import RouteExpansionSimulator
from src.geo_utils import load_airport_coordinates

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(layout="wide")
st.title("✈️ AeroNexus AI — Airline Strategy Intelligence Platform")

data_path = "data/US_DOT_Airfare_Historical_2008_2025.csv"

# -------------------------------------------------------
# Load Engines (Cached for Performance)
# -------------------------------------------------------
@st.cache_resource
def load_system():

    route_engine = RouteScoringEngine(data_path)
    route_engine.load_data()
    route_engine.compute_route_metrics()
    route_engine.generate_route_score()

    demand_engine = DemandPredictionEngine(data_path)
    demand_engine.load_data()
    demand_engine.train_models()

    network_engine = AirlineNetworkGraph(data_path)
    network_engine.load_data()
    network_engine.build_graph()

    simulator = RouteExpansionSimulator(
        graph=network_engine.graph,
        demand_model=demand_engine,
        route_df=route_engine.df
    )

    return route_engine, demand_engine, network_engine, simulator


route_engine, demand_engine, network_engine, simulator = load_system()
airport_coordinates = load_airport_coordinates()

# -------------------------------------------------------
# SECTION 1 — Strategic Hub Intelligence
# -------------------------------------------------------
st.subheader("🌍 Top Strategic Hub Airports")

top_hubs = network_engine.get_top_hubs(10)
st.dataframe(top_hubs, use_container_width=True)

# -------------------------------------------------------
# SECTION 2 — Route Expansion Intelligence
# -------------------------------------------------------
st.subheader("🔥 Top Route Expansion Recommendations")

top_routes = route_engine.get_top_routes(10)[
    ["origin_city", "destination_city", "route_score"]
]

st.dataframe(top_routes, use_container_width=True)

# -------------------------------------------------------
# SECTION 3 — AI Route Expansion Simulation
# -------------------------------------------------------
st.subheader("🚀 AI-Based New Route Expansion Opportunities")

expansion_results = simulator.simulate_new_routes(top_n=10)
st.dataframe(expansion_results, use_container_width=True)

# -------------------------------------------------------
# SECTION 4 — Interactive Geographic Route Simulator
# -------------------------------------------------------
st.subheader("🗺 Interactive Route Simulation Map")

airports = list(airport_coordinates.keys())

col1, col2 = st.columns(2)

with col1:
    origin = st.selectbox("Select Origin Airport", airports)

with col2:
    destination = st.selectbox("Select Destination Airport", airports)

if origin != destination:

    if origin in airport_coordinates and destination in airport_coordinates:

        lat1, lon1 = airport_coordinates[origin]
        lat2, lon2 = airport_coordinates[destination]

        fig = go.Figure()

        # Airport markers
        fig.add_trace(go.Scattergeo(
            lon=[lon1, lon2],
            lat=[lat1, lat2],
            text=[origin, destination],
            mode='markers',
            marker=dict(size=10),
        ))

        # Route line
        fig.add_trace(go.Scattergeo(
            lon=[lon1, lon2],
            lat=[lat1, lat2],
            mode='lines',
            line=dict(width=3),
            name="Route"
        ))

        fig.update_layout(
            geo=dict(
                scope='usa',
                projection_type='albers usa',
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
            ),
            showlegend=False,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------
        # Route Prediction Insights
        # -------------------------------------------------------
        distance = simulator._haversine(lat1, lon1, lat2, lon2)
        fare = simulator._estimate_fare(distance)

        avg_market_share = demand_engine.df[
            "largest_carrier_market_share"
        ].mean()

        features = pd.DataFrame([{
            "avg_fare": fare,
            "distance_miles": distance,
            "largest_carrier_market_share": avg_market_share
        }])

        predicted_passengers = demand_engine.models["Random Forest"].predict(features)[0]
        revenue = predicted_passengers * fare

        st.markdown("### 📊 Route Insights")

        colA, colB, colC = st.columns(3)

        colA.metric("Distance (miles)", f"{distance:,.0f}")
        colB.metric("Estimated Fare ($)", f"{fare:,.2f}")
        colC.metric("Predicted Revenue ($)", f"{revenue:,.0f}")

        # -------------------------------------------------------
        # Historical Route Trends (2008–2025)
        # -------------------------------------------------------
        st.markdown("## 📈 Historical Route Trends (2008–2025)")

        route_data = route_engine.df[
            (
                (route_engine.df["origin_city"] == origin) &
                (route_engine.df["destination_city"] == destination)
            ) |
            (
                (route_engine.df["origin_city"] == destination) &
                (route_engine.df["destination_city"] == origin)
            )
        ]

        if not route_data.empty:

            yearly = (
                route_data
                .groupby("year")
                .agg({
                    "passengers": "sum",
                    "avg_fare": "mean"
                })
                .reset_index()
                .sort_values("year")
            )

            yearly["revenue"] = yearly["passengers"] * yearly["avg_fare"]

            col_left, col_right = st.columns(2)

            # Passenger Trend
            with col_left:
                fig_pass = go.Figure()
                fig_pass.add_trace(go.Scatter(
                    x=yearly["year"],
                    y=yearly["passengers"],
                    mode="lines+markers"
                ))

                fig_pass.update_layout(
                    title="Passenger Trend (2008–2025)",
                    xaxis_title="Year",
                    yaxis_title="Passengers",
                    height=400
                )

                st.plotly_chart(fig_pass, use_container_width=True)

            # Revenue Trend
            with col_right:
                fig_rev = go.Figure()
                fig_rev.add_trace(go.Scatter(
                    x=yearly["year"],
                    y=yearly["revenue"],
                    mode="lines+markers"
                ))

                fig_rev.update_layout(
                    title="Revenue Trend (2008–2025)",
                    xaxis_title="Year",
                    yaxis_title="Revenue ($)",
                    height=400
                )

                st.plotly_chart(fig_rev, use_container_width=True)

        else:
            st.info("No historical data available for this route.")