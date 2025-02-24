import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.utils.model import load_model_and_scaler, make_prediction
from app.utils.data_processing import filter_data, group_by_hour
from app.utils.visualization import generate_graphs, generate_pie_chart, generate_gauge_chart

# Set page configuration
st.set_page_config(page_title="PowerBox Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("ETL/Clean_data/cleaned_solar_data.csv")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True, errors='coerce')  # Parse with dayfirst=True
    data = data.dropna(subset=['Timestamp'])  # Drop rows where parsing failed
    return data

solar_data = load_data()

# Load model and scaler
model, scaler = load_model_and_scaler()

# Sidebar filters
st.sidebar.header("Filters")
selected_profile = st.sidebar.selectbox("Select Customer Profile", solar_data['Customer Profile'].unique())
selected_panel_type = st.sidebar.selectbox("Select Solar Panels Type", solar_data['Solar Panels Type'].unique())
selected_month = st.sidebar.selectbox(
    "Select Month",
    options=[None] + list(range(1, 13)),
    format_func=lambda x: pd.to_datetime(f"2024-{x}-01").strftime("%B") if x else "All"
)

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "System Performance", "Statistics"])

# Overview Tab
with tab1:
    st.header("AI Predictions")
    
    # Input fields for all features
    avg_temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=50.0, value=25.0)
    solar_output = st.number_input("Solar Panels Energy Output (W)", min_value=0.0, max_value=2000.0, value=1000.0)
    battery_energy = st.number_input("Energy Stored in Batteries (kWh)", min_value=0.0, max_value=10.0, value=5.0)
    system_load = st.number_input("System Load (kW)", min_value=0.0, max_value=10.0, value=2.5)
    hour = st.number_input("Hour", min_value=0, max_value=23, value=14)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: pd.to_datetime(f"2024-{x}-01").strftime("%B"))
    day_of_week = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    is_weekend = st.radio("Is it a weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    
    # Create the input DataFrame
    input_features = pd.DataFrame([{
        "Temperature (°C)": avg_temperature,
        "Solar Panels Energy Output (W)": solar_output,
        "Energy Stored in Batteries (kWh)": battery_energy,
        "System Load (kW)": system_load,
        "Hour": hour,
        "Day": day,
        "Month": month,
        "DayOfWeek": day_of_week,
        "IsWeekend": is_weekend
    }])


    # Predict power consumption
    if st.button("Predict Power Consumption"):
        try:
            # Ensure feature names are retained
            predictions = make_prediction(model, scaler, input_features)
            st.success(f"Predicted Power Consumption: {predictions[0]:.2f} kWh")
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")




    # Data Filters Section
    filtered_data = filter_data(solar_data, selected_profile, selected_panel_type, selected_month)
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Group and Process Data
        hourly_data = group_by_hour(filtered_data)
        hourly_data['Datetime'] = pd.to_datetime(hourly_data['Date'].astype(str) + " " + hourly_data['Hour'].astype(str) + ":00:00")

        # Generate graphs
        fig_energy_output, fig_power_consumption, fig_battery_levels = generate_graphs(hourly_data)
        st.plotly_chart(fig_energy_output, use_container_width=True)
        st.plotly_chart(fig_power_consumption, use_container_width=True)
        st.plotly_chart(fig_battery_levels, use_container_width=True)



# System Performance Tab
with tab2:
    st.header("System Performance")
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Generate gauge
        avg_efficiency = filtered_data['Inverter Efficiency (%)'].mean()
        fig_efficiency = generate_gauge_chart(avg_efficiency, "Inverter Efficiency (%)", 100)
        st.plotly_chart(fig_efficiency, use_container_width=True)

        # Generate line chart
        fault_data = filtered_data.groupby('Timestamp').sum(numeric_only=True)['System Fault Alerts']
        fig_faults = px.line(x=fault_data.index, y=fault_data.values, title="System Fault Alerts Over Time", labels={'x': 'Timestamp', 'y': 'Fault Alerts'})
        st.plotly_chart(fig_faults, use_container_width=True)

# Statistics Tab
with tab3:
    st.header("Statistics")
    # Generate pie charts for distributions
    customer_dist = solar_data['Customer Profile'].value_counts()
    st.plotly_chart(generate_pie_chart(customer_dist, "Customer Profile Distribution"), use_container_width=True)

    panel_dist = solar_data['Solar Panels Type'].value_counts()
    st.plotly_chart(generate_pie_chart(panel_dist, "Solar Panels Type Distribution"), use_container_width=True)

    battery_tech_dist = solar_data['Battery Technology'].value_counts()
    st.plotly_chart(generate_pie_chart(battery_tech_dist, "Battery Technology Distribution"), use_container_width=True)




