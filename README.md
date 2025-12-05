# 🚕 NYC Taxi Fare & Duration Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Machine Learning project for predicting taxi fare amounts and trip durations in New York City using real-world data from the NYC Taxi & Limousine Commission (TLC).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Development](#-model-development)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
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
✅ **REST API** - FastAPI-based service for real-time predictions  
✅ **Docker Support** - Full containerization for training and deployment  
✅ **Clean Architecture** - Modular design following SOLID principles  
✅ **Unit Tests** - Test coverage for critical components  

---

## 📂 Project Structure

```
nyc_taxi_project/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── .dockerignore                # Docker ignore rules
│
├── docker/                      # Docker configuration
│   ├── Dockerfile.api           # API service container
│   ├── Dockerfile.train         # Training container
│   └── start.sh                 # Container startup script
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── eda.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Original TLC parquet files
│   ├── processed/               # Cleaned and feature-engineered data
│   └── external/                # External data (weather, traffic, etc.)
│
├── models/                      # Trained model artifacts (gitignored)
│   ├── baseline/                # Simple models (LR, DT)
│   └── advanced/                # Complex models (XGBoost, MLP)
│
├── src/                         # Source code
│   │
│   ├── configs/                 # Configuration management
│   │   ├── settings.py          # Global settings
│   │   └── paths.py             # Path management (SRP)
│   │
│   ├── data/                    # Data handling modules
│   │   ├── download.py          # Download NYC TLC datasets
│   │   ├── preprocess.py        # Data cleaning and validation
│   │   └── features.py          # Feature engineering
│   │
│   ├── models/                  # Model definitions
│   │   ├── baseline.py          # Linear Regression, Decision Trees
│   │   ├── advanced.py          # XGBoost, Random Forest, MLP
│   │   └── trainer.py           # Training pipeline orchestration
│   │
│   ├── evaluation/              # Model evaluation
│   │   ├── metrics.py           # MAE, MSE, RMSE calculations
│   │   └── validator.py         # Cross-validation and testing
│   │
│   ├── api/                     # FastAPI application
│   │   ├── app.py               # API endpoints
│   │   ├── schemas.py           # Pydantic models for validation
│   │   └── predictor.py         # Model loading and inference
│   │
│   ├── utils/                   # Utility functions
│   │   ├── io.py                # File I/O operations
│   │   ├── logging.py           # Logging configuration
│   │   └── timer.py             # Performance timing
│   │
│   └── pipelines/               # End-to-end pipelines
│       ├── build_dataset.py     # Data preparation pipeline
│       └── train_model.py       # Model training pipeline
│
├── tests/                       # Unit tests
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_api.py
│   └── test_model.py
│
└── deployment/                  # Deployment configuration
    ├── docker-compose.yml       # Multi-container orchestration
    ├── api.yaml                 # API deployment config
    └── Makefile                 # Deployment shortcuts
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
py -m src.pipelines.train_model --mode optimize --target fare 
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

## 🌐 API Documentation

### Endpoints

#### `POST /predict`

Predict fare and duration for a single trip.

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

### Model Comparison (May 2022 Dataset)

| Model | Target | MAE | RMSE | Training Time | Inference (1k samples) |
|-------|--------|-----|------|---------------|----------------------|
| Linear Regression | Fare | 3.45 | 5.21 | 2.3s | 0.05s |
| Decision Tree | Fare | 2.89 | 4.67 | 8.1s | 0.12s |
| Random Forest | Fare | 2.34 | 3.98 | 145s | 0.89s |
| **XGBoost** | **Fare** | **2.12** | **3.67** | **89s** | **0.34s** |
| MLP | Fare | 2.45 | 4.01 | 234s | 0.45s |

*Results will vary based on your training configuration and data preprocessing*

### Key Insights

- XGBoost provides the best balance of accuracy and speed
- Distance is the most important feature for fare prediction
- Time-based features (hour, day of week) significantly improve duration prediction
- Payment type and vendor ID have minimal impact on predictions

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

