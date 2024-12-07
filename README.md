# PowerBox ML Prediction Service

A machine learning service for predicting power consumption patterns using FastAPI.

## Project Structure

powerbox/
├── ML_powerbox_data/
│   ├── __init__.py
│   ├── api.py          # FastAPI endpoints for predictions
│   └── ML.py           # ML model and prediction logic
├── ETL/
│   └── Dashboard/
│       └── dashboard.py # Data visualization dashboard
├── models/             # Saved model files
└── requirements.txt
```

## Features

- Real-time power consumption predictions
- RESTful API endpoints using FastAPI
- Random Forest Regressor model
- Interactive dashboard for data visualization

## API Endpoints

### POST `/predict`
Predicts power consumption based on input parameters:
- temperature
- solar_output
- battery_energy
- system_load
- hour
- day
- month
- day_of_week
- is_weekend

### GET `/model-info`
Returns model metadata and features list

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the API server:

```bash
uvicorn ML_powerbox_data.api:app --reload
```

3. Access the API documentation:
```
http://localhost:8000/docs
```

## Example Request

```python
import requests

data = {
    "temperature": 25.0,
    "solar_output": 1000.0,
    "battery_energy": 5.0,
    "system_load": 2.5,
    "hour": 14,
    "day": 1,
    "month": 6,
    "day_of_week": 2,
    "is_weekend": 0
}

response = requests.post("http://localhost:8000/predict_consumption", json=data)
prediction = response.json()["predicted_consumption"]
```
