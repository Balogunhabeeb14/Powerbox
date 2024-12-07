from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from .ML import PowerConsumptionPredictor
from typing import List

app = FastAPI(title="Power Consumption Prediction API")

# Initialize the predictor and load the model
try:
    predictor = PowerConsumptionPredictor()
    predictor.load_model()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Initialize predictor without loading model for development
    predictor = PowerConsumptionPredictor()

class PredictionInput(BaseModel):
    temperature: float
    solar_output: float
    battery_energy: float
    system_load: float
    hour: int
    day: int
    month: int
    day_of_week: int
    is_weekend: int

class PredictionResponse(BaseModel):
    predicted_consumption: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_consumption(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([{
            'Temperature (°C)': input_data.temperature,
            'Solar Panels Energy Output (W)': input_data.solar_output,
            'Energy Stored in Batteries (kWh)': input_data.battery_energy,
            'System Load (kW)': input_data.system_load,
            'Hour': input_data.hour,
            'Day': input_data.day,
            'Month': input_data.month,
            'DayOfWeek': input_data.day_of_week,
            'IsWeekend': input_data.is_weekend
        }])
        
        # Make prediction
        prediction = predictor.predict(df)
        
        return PredictionResponse(predicted_consumption=float(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    return {
        "features": predictor.features,
        "model_type": "Random Forest Regressor",
        "version": "1.0"
    } 