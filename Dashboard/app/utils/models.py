import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("models/power_consumption_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler

def preprocess_inputs(input_df, scaler):
    # Model's expected features
    input_df = input_df[[
        "Temperature (°C)",
        "Solar Panels Energy Output (W)",
        "Energy Stored in Batteries (kWh)",
        "System Load (kW)",
        "Hour",
        "Day",
        "Month",
        "DayOfWeek",
        "IsWeekend"
    ]]
    
    # Scale the input features
    input_df_scaled = scaler.transform(input_df)
    return input_df_scaled


def make_prediction(model, scaler, input_df):
    # Preprocess inputs
    processed_inputs = preprocess_inputs(input_df, scaler)
    
    # DataFrame with correct column names
    feature_names = [
        "Temperature (°C)",
        "Solar Panels Energy Output (W)",
        "Energy Stored in Batteries (kWh)",
        "System Load (kW)",
        "Hour",
        "Day",
        "Month",
        "DayOfWeek",
        "IsWeekend"
    ]
    processed_inputs = pd.DataFrame(processed_inputs, columns=feature_names)

    # Make predictions
    predictions = model.predict(processed_inputs)
    return predictions

