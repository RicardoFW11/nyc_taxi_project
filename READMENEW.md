"""# 🚕 NYC Taxi Fare & Duration Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Machine Learning project for predicting taxi fare amounts and trip durations in New York City using real-world data from the NYC Taxi & Limousine Commission (TLC). This project demonstrates a complete MLOps lifecycle, from data ingestion to containerized deployment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Dataset](#-dataset)
- [Installation & Local Usage](#-installation--local-usage)
- [Docker Deployment](#-docker-deployment)
- [API Documentation](#-api-documentation)
- [Results](#-results)
- [Contributing](#-contributing)
- [Authors](#-authors)
- [License](#-license)

---

## 🎯 Overview

This project builds end-to-end machine learning pipelines to predict:

1. **Fare Amount** - The total cost of a taxi ride.
2. **Trip Duration** - The time duration of a trip in minutes.

The models leverage various features including:
- Pick-up and drop-off locations (latitude/longitude/zones).
- Date and time information (Hour, Day of Week).
- Vendor ID, passenger count, payment type.
- Engineered features (Euclidean distance, time-based traffic inference).

### 🏆 Project Goals

- Build **baseline models** (Linear Regression) for benchmarking.
- Develop **advanced models** (XGBoost) to capture non-linear relationships and edge cases.
- Deploy a **production-ready API** using FastAPI.
- Provide a user-friendly **Web Interface** using Streamlit.
- Ensure **reproducibility** through full Docker containerization.
- Follow **Clean Code** and **SOLID principles**.

---

## 🔄 System Architecture

The project follows a microservices architecture orchestrated by Docker:

```mermaid
graph LR
    A[Raw Data] --> B(Processing Pipeline)
    B --> C{Model Training}
    C -->|Option 1| D[Baseline: Linear Reg]
    C -->|Option 2| E[Advanced: XGBoost]
    D & E --> F[Model Artifacts .pkl]
    F --> G[FastAPI Backend]
    G --> H[Streamlit Frontend]



✨ Features
✅ End-to-End Pipeline - Automated data ingestion, cleaning, and feature engineering.
✅ Hybrid Modeling - Switch between Baseline (Linear) and Advanced (XGBoost) models with a single command. 
✅ REST API - FastAPI-based service for real-time predictions with Swagger UI. 
✅ Interactive UI - Streamlit app for easy user interaction and visualization. 
✅ Docker Ecosystem - Multi-container orchestration (Training -> API -> UI) using Docker Compose. 
✅ Robust Error Handling - Comprehensive validation using Pydantic schemas. 
✅ Clean Architecture - Modular design separating configuration, data, models, and presentation layers.


📂 Project Structure

nyc_taxi_project/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker orchestration
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Original TLC parquet files
│   └── processed/               # Cleaned and feature-engineered data
│
├── src/                         # Source code
│   │
│   ├── api/                     # FastAPI application
│   │   ├── app.py               # API endpoints
│   │   ├── schemas.py           # Pydantic models
│   │   └── predictor.py         # Inference logic
│   │
│   ├── config/                  # Configuration management
│   │   └── settings.py          # Global settings (Paths, Params)
│   │
│   ├── data/                    # Data handling modules
│   │   ├── preprocess.py        # Cleaning logic
│   │   └── features.py          # Feature engineering
│   │
│   ├── docker/                  # Dockerfiles
│   │   ├── Dockerfile.api       # API container
│   │   ├── Dockerfile.train     # Training container
│   │   └── Dockerfile.ui        # Streamlit container
│   │
│   ├── evaluation/              # Metrics calculation
│   │   └── metrics.py           # MAE, MSE, RMSE
│   │
│   ├── models/                  # Model definitions
│   │   ├── baseline.py          # Linear Regression wrapper
│   │   └── advanced.py          # XGBoost wrapper
│   │
│   ├── pipelines/               # Execution pipelines
│   │   ├── build_dataset.py     # ETL Pipeline
│   │   └── train_model.py       # Training Pipeline
│   │
│   └── streamlit_app.py         # Frontend application
│
└── models/                      # Trained model artifacts (gitignored)
    ├── baseline/                # Linear Regression artifacts
    └── advanced/                # XGBoost artifacts


🚀 Quick Start Guide
Option A: The "Docker Way" (Recommended) 🐳
Run the entire ecosystem (Training + API + UI) with a single command. No Python installation required.

```bash

docker-compose up --build

```

**Access the services:**

Web App (Frontend): http://localhost:8501
API Docs (Backend): http://localhost:8000/docs

Aquí tienes el script definitivo y completo. He verificado línea por línea que incluya todas las secciones que pediste (Features, Project Structure, Quick Start, Dataset, Installation, Docker, API, Results, Contributing, Authors y License).

Solo copia el siguiente bloque de código, pégalo en un archivo generate_readme.py y ejecútalo.

Python

import os

# Contenido COMPLETO del README.md
readme_content = r"""# 🚕 NYC Taxi Fare & Duration Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Machine Learning project for predicting taxi fare amounts and trip durations in New York City using real-world data from the NYC Taxi & Limousine Commission (TLC). This project demonstrates a complete MLOps lifecycle, from data ingestion to containerized deployment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Dataset](#-dataset)
- [Installation & Local Usage](#-installation--local-usage)
- [Docker Deployment](#-docker-deployment)
- [API Documentation](#-api-documentation)
- [Results](#-results)
- [Contributing](#-contributing)
- [Authors](#-authors)
- [License](#-license)

---

## 🎯 Overview

This project builds end-to-end machine learning pipelines to predict:

1. **Fare Amount** - The total cost of a taxi ride.
2. **Trip Duration** - The time duration of a trip in minutes.

The models leverage various features including:
- Pick-up and drop-off locations (latitude/longitude/zones).
- Date and time information (Hour, Day of Week).
- Vendor ID, passenger count, payment type.
- Engineered features (Euclidean distance, time-based traffic inference).

### 🏆 Project Goals

- Build **baseline models** (Linear Regression) for benchmarking.
- Develop **advanced models** (XGBoost) to capture non-linear relationships and edge cases.
- Deploy a **production-ready API** using FastAPI.
- Provide a user-friendly **Web Interface** using Streamlit.
- Ensure **reproducibility** through full Docker containerization.
- Follow **Clean Code** and **SOLID principles**.

---

## 🔄 System Architecture

The project follows a microservices architecture orchestrated by Docker:

```mermaid
graph LR
    A[Raw Data] --> B(Processing Pipeline)
    B --> C{Model Training}
    C -->|Option 1| D[Baseline: Linear Reg]
    C -->|Option 2| E[Advanced: XGBoost]
    D & E --> F[Model Artifacts .pkl]
    F --> G[FastAPI Backend]
    G --> H[Streamlit Frontend]
✨ Features
✅ End-to-End Pipeline - Automated data ingestion, cleaning, and feature engineering. ✅ Hybrid Modeling - Switch between Baseline (Linear) and Advanced (XGBoost) models with a single command. ✅ REST API - FastAPI-based service for real-time predictions with Swagger UI. ✅ Interactive UI - Streamlit app for easy user interaction and visualization. ✅ Docker Ecosystem - Multi-container orchestration (Training -> API -> UI) using Docker Compose. ✅ Robust Error Handling - Comprehensive validation using Pydantic schemas. ✅ Clean Architecture - Modular design separating configuration, data, models, and presentation layers.

📂 Project Structure
Plaintext

nyc_taxi_project/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker orchestration
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Original TLC parquet files
│   └── processed/               # Cleaned and feature-engineered data
│
├── src/                         # Source code
│   │
│   ├── api/                     # FastAPI application
│   │   ├── app.py               # API endpoints
│   │   ├── schemas.py           # Pydantic models
│   │   └── predictor.py         # Inference logic
│   │
│   ├── config/                  # Configuration management
│   │   └── settings.py          # Global settings (Paths, Params)
│   │
│   ├── data/                    # Data handling modules
│   │   ├── preprocess.py        # Cleaning logic
│   │   └── features.py          # Feature engineering
│   │
│   ├── docker/                  # Dockerfiles
│   │   ├── Dockerfile.api       # API container
│   │   ├── Dockerfile.train     # Training container
│   │   └── Dockerfile.ui        # Streamlit container
│   │
│   ├── evaluation/              # Metrics calculation
│   │   └── metrics.py           # MAE, MSE, RMSE
│   │
│   ├── models/                  # Model definitions
│   │   ├── baseline.py          # Linear Regression wrapper
│   │   └── advanced.py          # XGBoost wrapper
│   │
│   ├── pipelines/               # Execution pipelines
│   │   ├── build_dataset.py     # ETL Pipeline
│   │   └── train_model.py       # Training Pipeline
│   │
│   └── streamlit_app.py         # Frontend application
│
└── models/                      # Trained model artifacts (gitignored)
    ├── baseline/                # Linear Regression artifacts
    └── advanced/                # XGBoost artifacts
🚀 Quick Start Guide
Option A: The "Docker Way" (Recommended) 🐳
Run the entire ecosystem (Training + API + UI) with a single command. No Python installation required.

```bash
docker-compose up --build
```

Access the services:

Web App (Frontend): http://localhost:8501
API Docs (Backend): http://localhost:8000/docs

📊 Dataset

NYC TLC Trip Record Data (2022)
We use the Yellow Taxi Trip Records from the NYC Taxi & Limousine Commission:
Source: NYC TLC Trip Record Data
Format: Parquet files.
Start Date: May 2022.
Key Features Used: VendorID, passenger_count, trip_distance, pickup_hour, day_of_week, payment_type.

💻 Installation & Local Usage

If you prefer to run the project without Docker for development purposes:

**1. Environment Setup**

# Clone the repository

```bash
git clone [https://github.com/yourusername/nyc_taxi_project.git](https://github.com/yourusername/nyc_taxi_project.git)
cd nyc_taxi_project
```

# Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
```

# Install Dependencies

```bash
pip install -r requirements.txt
```


2. Data Preparation Pipeline
Loads raw Parquet data, cleans outliers, and generates train_data.parquet.

```bash
python src/pipelines/build_dataset.py
```

**3. Model Training**

You can choose between Baseline and Advanced models via command line arguments.

Train Baseline (Linear Regression):

```bash
python src/pipelines/train_model.py --model linear
Output: Saves to models/baseline/linear_fare.pkl
```

Train Advanced (XGBoost):

```bash
python src/pipelines/train_model.py --model xgboost
Output: Saves to models/advanced/xgboost_fare.pkl
```

4. Running Services Locally
Start the API:

```bash
uvicorn src.api.app:app --reload
```

Start the Streamlit UI:

```bash
# In a new terminal
streamlit run src/streamlit_app.py
```


🐳 Docker Deployment Details
The project uses Docker Compose to orchestrate three containers:

· train_xgboost:

    Executes the training pipeline automatically upon startup.
    Mounts a volume to persist the trained model (.pkl) to your host machine.
    Exits automatically after training completes.

· api:

    Starts the FastAPI server.
    Waits for the model file to be available (via volume sharing).
    Exposes port 8000.

· ui:

    Starts the Streamlit application.
    Communicates with the API container via the internal Docker network (http://api:8000).
    Exposes port 8501.

## 🌐 API Documentation

### Endpoints

#### `POST /predict`

Predict fare for a single trip.

**Request Body Example:**

```json
{
  "VendorID": 1,
  "passenger_count": 2,
  "trip_distance": 5.0,
  "payment_type": 1,
  "pickup_datetime": "2022-05-15 16:00:00"
}
```

**Response Example:**
JSON
```json
{
  "predicted_fare": 20.66,
  "model_version": "xgboost_fare_v1",
  "prediction_timestamp": "2025-12-06T13:44:03",
  "input_features": {
    "pickup_hour": 16,
    "pickup_day_of_week": 6,
    "distance_euclidean": 5.0
  }
}
```

#### GET /health

Check API health and model loading status.

#### `GET /models`

List available models and their metadata.

### Using the Web Interface

The Streamlit app provides an intuitive interface for making predictions:

1. **Health Check**: Visual indicator shows if API is online (🟢 green)
2. **Input Form**: Easy-to-use form with dropdowns and number inputs
3. **Instant Predictions**: Click "Predict Fare" to get results
4. **Detailed Results**: View model version, engineered features, and technical details
5. **Example Trips**: Quick buttons to test common trip scenarios

**Key Features:**
- 🎨 Beautiful, responsive UI
- 📊 Real-time predictions
- 🔍 Technical details for debugging
- 💡 Example trips for quick testing
- ⚡ Fast and lightweight

## 📈 Results

**Dataset Statistics:**
- Total records loaded: 3,588,295
- Sample size: 100,000 rows
- After cleaning: 96,406 rows
- Train set: 77,124 samples (80%)
- Test set: 19,282 samples (20%)

**Linear Regression Baseline:**

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | $2.66 |
| MSE (Mean Squared Error) | 42.62 |
| RMSE (Root Mean Squared Error) | $6.53 |
| Training Time | ~1.2s |
| Model Size | 3 KB |

### Interpretation

- **Average Error**: The model predicts fares with an average error of **$2.66**
- **Typical Error Range**: 68% of predictions fall within ±$6.53 of actual fare
- **Performance**: Good baseline performance for real-time predictions
- **Speed**: Fast training and inference suitable for production use


## Model Performance Comparison

Metric     Linear Regression (Baseline)     XGBoost (Advanced)
MAE             ~$2.66,                         **~$2.03**
RMSE            ~$6.53                          **~$5.75**

**Key Insights from Deployment**

    · XGBoost Superiority: The advanced model significantly outperformed the baseline, reducing the Mean Absolute Error (MAE) by approximately 23%.

    · Temporal Logic: The model successfully learned time-based pricing patterns. For example, a 5-mile trip at 4:00 PM (Traffic) costs ~$20.66, while the same trip at 4:00 AM (No Traffic) drops to ~$17.97.

    · Minimum Fare: The model correctly predicts minimum fares for short trips (e.g., 0.1 miles ≈ $5.51), respecting NYC regulation pricing (base fare + surcharges).


### Next Steps for Improvement

Future model iterations could explore:
- **Advanced Models**: XGBoost, Random Forest, Neural Networks  --> semicompleted
- **More Features**: Weather data, traffic patterns, holidays
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Full Dataset**: Training on complete 3.5M records --> completed
- **Feature Selection**: Identify and remove low-impact features

---

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test suite:

```bash
pytest tests/test_api.py -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 References

### Papers and Articles

- [Fare and Duration Prediction: A Study of New York City Taxi Rides](https://www.researchgate.net/publication/335332532_Fare_and_Duration_Prediction_A_Study_of_New_York_City_Taxi_Rides)
- [Towards Data Science - NYC Taxi Fare Prediction](https://towardsdatascience.com/tagged/nyc-taxi)
- [NYC Yellow Taxi Demand Prediction using ML](https://arxiv.org/abs/2004.14419)

### Official Documentation

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Trip Record User Guide](https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf)
- [Taxi Zone Shapefile](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc)


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

    1. Fork the repository.
    2. Create a feature branch (git checkout -b feature/AmazingFeature).
    3. Commit your changes (git commit -m 'Add some AmazingFeature').
    4. Push to the branch (git push origin feature/AmazingFeature).
    5. Open a Pull Request.

## 👥 Authors
Ricardo Walters - Initial Project Architect
Alex Sanchez - ML Developer Career Project 
Patricia Roman - ML Developer Career Project 
Ronny Moreno - ML Developer Career Project 
Fernando Wuy - ML Developer Career Project 
Ricardo Torres - ML Developer Career Project 
Kevin Peña - ML Developer Career Project 

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Built with ❤️ using Python, FastAPI, XGBoost, Streamlit and Docker """