https://aeronexus-ai.up.railway.app/

✈️ AeroNexus AI — Airline Strategy Intelligence Platform
🚀 Overview

A scalable airline strategy intelligence platform built using Machine Learning, Network Analysis, and Cloud Deployment.

The system analyzes historical U.S. DOT airfare data (2008–2025) to perform route scoring, hub optimization, demand forecasting, and AI-driven route expansion simulation in real time.

This project demonstrates:

End-to-end ML system design

Network graph analytics

Predictive demand modeling

Geospatial computation

Real-time interactive dashboards

Containerized cloud deployment

CI/CD pipeline integration

🏗 Architecture

U.S. DOT Airfare Dataset (2008–2025)
↓
Data Cleaning & Feature Engineering
↓
Route Scoring Engine
↓
Demand Prediction Models (Linear Regression + Random Forest)
↓
Network Graph Analysis (NetworkX Centrality Metrics)
↓
Route Expansion Simulation Engine
↓
Streamlit + Plotly Interactive Dashboard
↓
Docker Containerization
↓
CI/CD Deployment on Railway Cloud

🛠 Tech Stack

Python

Scikit-learn

XGBoost (if included)

NetworkX

Pandas & NumPy

Streamlit

Plotly

Docker

Docker Hub

Railway Cloud

Git & GitHub

📡 Data Source

U.S. Department of Transportation (DOT) Airfare Dataset

Includes:

Origin City

Destination City

Distance (miles)

Passenger Volume

Average Fare

Largest Carrier Market Share

Historical Quarterly Data (2008–2025)

Total:

68,000+ cleaned records

1,600+ airline routes

169 airports

⚙️ Features
1️⃣ Route Intelligence Scoring

Aggregates historical revenue per route

Computes revenue volatility

Calculates growth rates

Ranks high-revenue, stable expansion routes

Identifies top-performing airline corridors

2️⃣ Machine Learning Demand Forecasting

Feature engineering using distance, fare, and market share

Linear Regression baseline model

Random Forest Regressor for nonlinear modeling

Model comparison using RMSE and R²

Passenger demand prediction for new and existing routes

3️⃣ Airline Network Graph Analysis

Airports modeled as nodes

Routes modeled as edges

Degree centrality for connectivity ranking

Betweenness centrality for strategic hub identification

169 airports analyzed using graph-based metrics

4️⃣ Route Expansion Simulation Engine

Simulates origin–destination pairs not currently connected

Computes geographic distance using Haversine formula

Estimates fare using distance-based pricing logic

Predicts passenger demand via trained ML model

Projects route-level revenue

Outputs:

Distance

Estimated Fare

Predicted Passengers

Projected Revenue

5️⃣ Interactive Geospatial Dashboard

Real-time route selection via dropdown

U.S. map visualization using Plotly Scattergeo

Dynamic route line rendering

KPI metrics display

Historical passenger trend (2008–2025)

Historical revenue trend (2008–2025)

Real-time simulation insights

6️⃣ Containerized Cloud Deployment

Full application containerized using Docker

Image published to Docker Hub

CI/CD-enabled deployment pipeline

Hosted on Railway cloud infrastructure

Reproducible, portable runtime environment

Production-ready architecture

🔒 System Capabilities

✔ Scalable ML architecture
✔ Graph-based network intelligence
✔ Geospatial analytics
✔ Cloud-native deployment
✔ Containerized reproducibility
✔ Interactive decision-support system

🎯 Outcome

AeroNexus AI transforms static airline historical data into an interactive, predictive strategy intelligence platform that enables data-driven route expansion and hub optimization decisions.
