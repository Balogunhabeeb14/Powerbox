import plotly.express as px
import plotly.graph_objects as go

def generate_graphs(hourly_data):
    """
    Generate time-series graphs for energy output, power consumption, and battery levels.
    """
    energy_output = px.line(hourly_data, x="Datetime", y="Solar Panels Energy Output (W)", title="Hourly Solar Panels Energy Output")
    power_consumption = px.line(hourly_data, x="Datetime", y="Power Consumption (kW)", title="Hourly Power Consumption")
    battery_levels = px.line(hourly_data, x="Datetime", y="Energy Stored in Batteries (kWh)", title="Hourly Battery Levels")
    return energy_output, power_consumption, battery_levels

def generate_pie_chart(data, title):
    """
    Generate a pie chart for distribution analysis.
    """
    return px.pie(values=data.values, names=data.index, title=title)

def generate_gauge_chart(value, title, max_value):
    """
    Generate a gauge chart for percentage metrics.
    """
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if value is not None else 0,
        title={'text': title},
        gauge={'axis': {'range': [None, max_value]}}
    ))
