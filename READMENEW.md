# NYC Taxi Fare & Duration Intelligence System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Executive Summary

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict taxi fare amounts and trip durations in New York City. By leveraging high-volume transactional data from the NYC Taxi & Limousine Commission (TLC), the system provides real-time inference capabilities through a microservices architecture.

The solution encompasses the entire data lifecycle: ingestion of raw parquet files, automated cleaning and validation, feature engineering, model training (benchmarking Linear Regression vs. XGBoost), and deployment via containerized REST APIs and interactive dashboards.

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Project Features](#-project-features)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Data Pipeline & Engineering](#-data-pipeline--engineering)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Deployment Strategy](#-deployment-strategy)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 🔄 System Architecture

The solution relies on a containerized microservices pattern managed by Docker Compose, ensuring isolation and reproducibility across environments.

```mermaid
graph LR
    subgraph Data Layer
    A[Raw TLC Data] --> B(Preprocessing Pipeline)
    B --> C(Feature Engineering)
    end
    
    subgraph Training Layer
    C --> D{Model Selection}
    D -->|Baseline| E[Linear Regression]
    D -->|Advanced| F[XGBoost / Random Forest]
    E & F --> G[Serialized Artifacts .pkl]
    end
    
    subgraph Serving Layer
    G --> H[FastAPI Backend]
    H --> I[Streamlit Dashboard]
    end

✨ Project Features

Core Capabilities

* Automated ETL Pipeline: Robust extraction, transformation, and loading process capable of handling millions of records with memory-efficient sampling.

* Dual-Target Prediction: Simultaneous modeling of financial metrics (Fare Amount) and operational metrics (Trip Duration).

* Hybrid Modeling Strategy: Implementation of baseline models for benchmark establishment and advanced ensemble methods (XGBoost) for high-precision inference.

* Production-Ready API: High-performance REST interface built with FastAPI, including Pydantic validation and Swagger documentation.

* Interactive Analytics Dashboard: User-facing interface developed in Streamlit for real-time scenario testing and model explainability.

Technical Standards
Containerization: Full Docker support for development and deployment.

Clean Architecture: Separation of concerns between configuration, data processing, modeling, and presentation layers.

Code Quality: Type hinting, modular design, and comprehensive logging.

📂 Repository Structure

nyc_taxi_project/
│
├── data/                        # Local data storage (GitIgnored)
│   ├── raw/                     # Raw TLC parquet files
│   └── processed/               # Cleaned and engineered datasets
│
├── models/                      # Serialized model artifacts (GitIgnored)
│   ├── baseline/                # Linear Regression & Decision Trees
│   └── advanced/                # XGBoost & Random Forest models
│
├── src/                         # Application Source Code
│   ├── api/                     # Backend Service
│   │   ├── app.py               # Main application entry point
│   │   ├── schemas.py           # Data validation models
│   │   └── predictor.py         # Inference engine wrapper
│   │
│   ├── config/                  # Configuration Management
│   │   ├── settings.py          # Environment variables & constants
│   │   └── paths.py             # Directory structure definitions
│   │
│   ├── data/                    # ETL Pipeline Modules
│   │   ├── download.py          # Data ingestion
│   │   ├── preprocess.py        # Data cleaning & validation logic
│   │   ├── features.py          # Feature engineering pipeline
│   │   └── data_splitter.py     # Train/Test/Val stratification
│   │
│   ├── evaluation/              # Model Assessment
│   │   └── metrics.py           # Standardized regression metrics
│   │
│   ├── models/                  # Estimator Definitions
│   │   ├── base_model.py        # Abstract Base Class
│   │   ├── baseline.py          # Sklearn implementations
│   │   └── advanced.py          # Gradient Boosting implementations
│   │
│   ├── pipelines/               # Execution Orchestrators
│   │   ├── build_dataset.py     # Full ETL execution script
│   │   └── train_model.py       # Training & Optimization script
│   │
│   ├── docker/                  # Container Definitions
│   │   ├── Dockerfile.api       # REST API Image
│   │   ├── Dockerfile.train     # Training Job Image
│   │   └── Dockerfile.ui        # Dashboard Image
│   │
│   └── streamlit_app.py         # Frontend application
│
├── docker-compose.yml           # Service orchestration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

🚀 Quick Start Guide
Option A: Containerized Deployment (Recommended)
Deploy the entire ecosystem (Training Job + API + UI) in a single step using Docker Compose. This ensures all dependencies are isolated.

docker-compose up --build

Service Access Points:

Frontend (Streamlit): http://localhost:8501

API Documentation (Swagger): http://localhost:8000/docs

Option B: Local Development
For developers wishing to debug or extend the code locally.

Environment Setup


python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt


Execute Data PipelineDownloads raw data and generates the training set.Bashpython src/pipelines/build_dataset.py
Train ModelsTrain the advanced XGBoost model.Bashpython src/pipelines/train_model.py --model xgboost
Run ServicesBash# Terminal 1: Start API
uvicorn src.api.app:app --reload

# Terminal 2: Start Dashboard
streamlit run src/streamlit_app.py
📊 Data Pipeline & EngineeringThe system utilizes NYC Yellow Taxi Trip Records (May 2022).Preprocessing StrategyTemporal Cleaning: Filtering of future dates and anomalous trip durations (e.g., > 3 hours or < 1 minute).Geospatial Validation: Exclusion of coordinates outside NYC operational zones.Financial Consistency: Validation of total amounts against the sum of fare components (surcharges, taxes, tolls).Feature EngineeringThe model transforms raw transactional data into predictive features:Temporal Features: Hour of day, day of week, rush-hour indicators.Spatial Features: Pickup/Dropoff boroughs, airport proximity flags (JFK, LGA, EWR).Operational Features: Euclidean distance estimates, traffic-based speed inferences.📈 Model PerformanceThe system evaluates models using a Hold-out validation strategy (80% Train, 20% Test).MetricLinear Regression (Baseline)XGBoost (Advanced)ImprovementMAE (Mean Absolute Error)~$2.66**~$2.03**23.7%RMSE (Root Mean Squared Error)~$6.53**~$5.75**11.9%Key Insights:Non-Linearity: The XGBoost model successfully captures non-linear pricing dynamics, such as traffic congestion effects during rush hours.Minimum Fare Logic: The advanced model correctly predicts the statutory minimum fare for short trips, whereas linear models tend to underestimate.🌐 API ReferenceHealth CheckGET /healthReturns the operational status of the API and loaded models.Prediction EndpointPOST /predictGenerates fare and duration estimates for a single trip configuration.Request Payload:JSON{
  "VendorID": 1,
  "passenger_count": 1,
  "trip_distance": 3.5,
  "payment_type": 1,
  "pickup_datetime": "2022-05-15 14:30:00"
}
Response Object:JSON{
  "predicted_fare": 15.50,
  "predicted_duration": 18.0,
  "model_version": "xgboost_v1.2",
  "confidence_score": 92.5
}
🐳 Deployment StrategyThe application is containerized into three distinct services:train_xgboost: Ephemeral container. Runs the ETL and training pipeline on startup, saves artifacts to a shared volume, and terminates.api: Persistent container. Waits for model artifacts to be available, then launches the Uvicorn server.ui: Persistent container. Hosts the Streamlit frontend and communicates with the API service via the internal Docker network.



👥 Authors & Acknowledgments
Project Architects:

Ricardo Walters

Alex Sanchez

Patricia Roman

Ronny Moreno

Fernando Wuy

Ricardo Torres

Kevin Peña

This project was developed as part of the Anyone AI Machine Learning Developer Career, demonstrating proficiency in full-stack Machine Learning Engineering.