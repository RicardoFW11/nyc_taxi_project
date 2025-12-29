# 🚕 NYC Taxi Fare & Duration Prediction

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
- [Installation](#-installation)
- [Usage (Local)](#-usage-local)
- [Docker Deployment](#-docker-deployment)
- [API Documentation](#-api-documentation)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)


---

## 🎯 Overview

This project builds end-to-end machine learning pipelines to predict:

1. **Fare Amount** - The total cost of a taxi ride
2. **Trip Duration** - The time duration of a trip in minutes

The models leverage various features including:
- Pick-up and drop-off locations (latitude/longitude)
- Date and time information
- Vendor ID, passenger count, payment type
- Engineered features (distance, time-based features, etc.)
- *(Optional)* External data such as weather conditions and traffic


### 🔄 System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Raw Data   │────▶│  Processing  │─────▶│   Trained   │
│ (Parquet)   │      │   Pipeline   │      │    Model    │
└─────────────┘      └──────────────┘      └──────┬──────┘
                                                    │
                                                    ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Streamlit  │◀─────│   FastAPI    │◀─────│   Model     │
│  Frontend   │      │     API      │      │  Predictor  │
└─────────────┘      └──────────────┘      └─────────────┘
     (UI)              (Backend)              (Inference)
```

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

### 🏆 Project Goals

- Build **baseline models** (Linear Regression, Decision Trees)
- Develop **advanced models** (XGBoost, Random Forest, Neural Networks)
- Compare model performance using multiple metrics
- Deploy a **production-ready API** for real-time predictions
- Ensure **reproducibility** through Docker containerization
- Follow **Clean Code** and **SOLID principles**

---

## ✨ Features

✅ **Comprehensive EDA** - Jupyter notebooks with in-depth data analysis  
✅ **Feature Engineering** - Distance calculation, temporal features, zone mapping  
✅ **Multiple Models** - Baseline and advanced ML algorithms  
✅ **Model Comparison** - Systematic evaluation of MAE, MSE, RMSE, training/inference time  
✅ **REST API** - FastAPI-based service for real-time predictions with Swagger UI.  
✅ **Interactive UI** - Streamlit app for easy user interaction and visualization.  
✅ **Docker Ecosystem** - Multi-container orchestration (Training -> API -> UI) using Docker Compose.  
✅ **Clean Architecture** - Modular design separating configuration, data, models, and presentation layers.  
✅ **Unit Tests** - Test coverage for critical components  
✅ **End-to-End Pipeline** - Automated data ingestion, cleaning, and feature engineering. 
✅ **Hybrid Modeling** - Switch between Baseline (Linear) and Advanced (XGBoost) models with a single flag. 
✅ **Robust Error Handling** - Comprehensive validation using Pydantic schemas. 


---

## 📂 Project Structure

```
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

```

### 🧩 Architecture Design Principles

This project follows industry best practices:

| Principle | Implementation |
|-----------|----------------|
| **Single Responsibility** | Each module has one clear purpose (e.g., `preprocess.py` only cleans data) |
| **Open/Closed** | New models can be added without modifying existing code |
| **Liskov Substitution** | All models expose consistent `fit()` and `predict()` interfaces |
| **Interface Segregation** | API, training, and data processing are independent |
| **Dependency Inversion** | Configuration and paths are externalized, not hardcoded |

---

## 📊 Dataset

### NYC TLC Trip Record Data (2022)

We use the **Yellow Taxi Trip Records** from the NYC Taxi & Limousine Commission:

- **Source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Format**: Parquet files (one per month)
- **Recommended Start**: May 2022 (~3M records)
- **Features**: 19 columns including pickup/dropoff coordinates, timestamps, fare amounts, passenger count, etc.

### Data Dictionary

Key features used in this project:

| Feature | Description |
|---------|-------------|
| `VendorID` | Provider ID (1=Creative Mobile, 2=VeriFone) |
| `tpep_pickup_datetime` | Pick-up date and time |
| `tpep_dropoff_datetime` | Drop-off date and time |
| `passenger_count` | Number of passengers |
| `trip_distance` | Trip distance in miles |
| `pickup_longitude/latitude` | GPS coordinates of pickup |
| `dropoff_longitude/latitude` | GPS coordinates of dropoff |
| `fare_amount` | **Target variable** - fare in USD |
| `payment_type` | Payment method (1=Credit, 2=Cash, etc.) |

**Useful Resources**:
- [Trip Record User Guide](https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf)
- [Yellow Trips Data Dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
- [Taxi Zone Maps and Lookup Tables](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Docker (optional, for containerized deployment)
- Git

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/nyc_taxi_project.git
cd nyc_taxi_project
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Download the dataset**

```bash
python src/data/download.py --month 2022-05
```

---

## 💻 Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Data Preparation

```bash
python src/pipelines/build_dataset.py
```

This will:
- Load raw data from `data/raw/`
- Clean and validate the data
- Engineer features
- Save processed data to `data/processed/`

### 3. Model Training

```bash
python src/pipelines/train_model.py --model xgboost --target fare_amount
```

Options:
- `--model`: `linear`, `decision_tree`, `random_forest`, `xgboost`, `mlp`
- `--target`: `fare_amount`, `trip_duration`, or `both`

### 4. Model Evaluation

```bash
python src/evaluation/validator.py --model-path models/xgboost_fare.pkl
```

### 5. Run API Server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: `http://localhost:8000/docs`

### 6. Run Streamlit Web App

**Option A: Using the Web Interface (Recommended for demos)**

In a new terminal, run:

```bash
streamlit run streamlit_app.py
```

Then open your browser at: `http://localhost:8501`

The web interface allows you to:
- ✅ Fill in trip details using intuitive forms
- ✅ Get instant fare predictions with one click
- ✅ View prediction details and technical information
- ✅ Try example trips (short, medium, long distances)

**Option B: Using the API directly**

For programmatic access, make POST requests to the API:

```python
import requests

trip_data = {
    "VendorID": 1,
    "passenger_count": 2,
    "trip_distance": 5.3,
    "payment_type": 1,
    "pickup_datetime": "2022-05-15 14:30:00"
}

response = requests.post(
    "http://localhost:8000/predict",
    json=trip_data
)

print(response.json())
# Output: {'predicted_fare': 18.45, 'model_version': 'linear_fare_v1', ...}
```

---

## 🧠 Model Development

### Baseline Models

1. **Linear Regression** - Fast, interpretable baseline
2. **Decision Tree** - Non-linear relationships

### Advanced Models

1. **Random Forest** - Ensemble of decision trees
2. **XGBoost** - Gradient boosting with regularization
3. **LightGBM** - Fast gradient boosting
4. **Multi-Layer Perceptron (MLP)** - Neural network approach

### Evaluation Metrics

- **MAE** (Mean Absolute Error) - Average prediction error
- **MSE** (Mean Squared Error) - Penalizes large errors
- **RMSE** (Root Mean Squared Error) - Same units as target
- **Training Time** - Model training duration
- **Inference Time** - Prediction speed

---

## 🚀 Quick Start Guide

### Option A: The "Docker Way" 🐳

**Run the entire ecosystem (Training + API + UI) with a single command. No Python installation required.**

```bash
# Build the docker ecosystem
docker-compose up --build
```

#Access the services:

Web App: http://localhost:8501
API Docs: http://localhost:8000/docs


**Option B: The "Local Way" (For Development) 💻**

**1. Create Environment**
```bash    
    python -m venv venv

    # Windows:
    .\venv\Scripts\Activate

    # Mac/Linux:

    source venv/bin/activate
    pip install -r requirements.txt


**2. Prepare Data & Train**

    # Download and process data
    python src/pipelines/build_dataset.py

    # Train Advanced Model (XGBoost)
    python src/pipelines/train_model.py --model xgboost

**3. Run Services**

    # Terminal 1: Start API
    uvicorn src.api.app:app --reload

    # Terminal 2: Start UI
    streamlit run src/streamlit_app.py
  ```    
**🎉 Done!** Open your browser at `http://localhost:8501` and start predicting fares!

---

## 🌐 API Documentation

### Endpoints

#### `POST /predict`

Predict  fare and duration for a single trip.

**Request Body**:
```json
{
  "pickup_datetime": "2022-05-15T14:30:00",
  "pickup_longitude": -73.9851,
  "pickup_latitude": 40.7589,
  "dropoff_longitude": -73.9683,
  "dropoff_latitude": 40.7854,
  "passenger_count": 2,
  "vendor_id": 1,
  "payment_type": 1
}
```

**Response**:
```json
{
  "predicted_fare": 12.50,
  "predicted_duration": 15.3,
  "model_version": "xgboost_v1.0",
  "timestamp": "2024-11-28T10:30:00"
}
```

#### `GET /health`

Check API health status.

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

---

## 🐳 Docker Deployment

### Build and Run Training Container

```bash
docker build -f docker/Dockerfile.train -t nyc-taxi-train .
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models nyc-taxi-train
```

### Build and Run API Container

```bash
docker build -f docker/Dockerfile.api -t nyc-taxi-api .
docker run -p 8000:8000 nyc-taxi-api
```

### Using Docker Compose

```bash
cd deployment
docker-compose up -d
```

This will start:
- API service on port 8000
- (Optional) Database for storing predictions
- (Optional) Monitoring dashboard

---

## 📈 Results

### Model Performance (May 2022 Dataset - 100k Sample)

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

### Next Steps for Improvement

Future model iterations could explore:
- **Advanced Models**: XGBoost, Random Forest, Neural Networks
- **More Features**: Weather data, traffic patterns, holidays
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Full Dataset**: Training on complete 3.5M records
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

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation as needed

---

## 👥 Authors

**Ricardo** - ML Developer Career Project

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for providing the dataset
- ML Developer Career program for project guidance
- Open-source ML community for tools and frameworks

---

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Python, FastAPI, XGBoost, and Docker**

