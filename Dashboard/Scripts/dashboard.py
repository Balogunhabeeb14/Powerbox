import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load and prepare your data
solar_data = pd.read_csv('ETL/Clean_data/cleaned_solar_data.csv')

# Convert 'Timestamp' column to datetime
solar_data['Timestamp'] = pd.to_datetime(solar_data['Timestamp'])

# Filter the data based on selected profile, panel type, and month
def filter_data(profile, panel_type, month):
    filtered_data = solar_data[
        (solar_data['Customer Profile'] == profile) &
        (solar_data['Solar Panels Type'] == panel_type)
    ]
    
    # Filter data by month if specified
    if month is not None:
        filtered_data = filtered_data[filtered_data['Timestamp'].dt.month == month]
    
    return filtered_data

# Group data by hour
def group_by_hour(filtered_data):
    # Create a new column for hour
    filtered_data['Hour'] = filtered_data['Timestamp'].dt.hour
    filtered_data['Date'] = filtered_data['Timestamp'].dt.date
    
    # Group by Date and Hour and calculate the mean of numeric columns
    return filtered_data.groupby(['Date', 'Hour']).mean(numeric_only=True).reset_index()

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='PowerBox System Dashboard'),

    html.Div(children='''Interactive dashboard for visualizing solar energy system performance.'''),

    # Tabs for different sections of the dashboard
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.Label('Select Customer Profile'),
                dcc.Dropdown(
                    id='customer-profile-dropdown',
                    options=[{'label': profile, 'value': profile} for profile in solar_data['Customer Profile'].unique()],
                    value=solar_data['Customer Profile'].unique()[0],  # Default value
                    clearable=False
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.Label('Select Solar Panels Type'),
                dcc.Dropdown(
                    id='solar-panels-type-dropdown',
                    options=[{'label': panel_type, 'value': panel_type} for panel_type in solar_data['Solar Panels Type'].unique()],
                    value=solar_data['Solar Panels Type'].unique()[0],  # Default value
                    clearable=False
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.Label('Select Month'),
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[
                        {'label': month, 'value': month_num} 
                        for month_num, month in enumerate(['January', 'February', 'March', 'April', 'May', 'June',
                                                            'July', 'August', 'September', 'October', 'November', 'December'], 1)
                    ],
                    value=None,  # Default value
                    clearable=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            # Main visuals in this tab
            dcc.Graph(id='energy-output-graph'),
            dcc.Graph(id='power-consumption-graph'),
            dcc.Graph(id='battery-levels-graph'),
        ]),

        dcc.Tab(label='System Performance', children=[
            dcc.Graph(id='system-load-voltage-graph'),
            dcc.Graph(id='inverter-efficiency-gauge'),
            dcc.Graph(id='system-faults-graph'),
        ]),

        dcc.Tab(label='Statistics', children=[
            dcc.Graph(id='customer-distribution-pie'),
            dcc.Graph(id='solar-panels-type-pie'),
            dcc.Graph(id='battery-technology-pie'),
        ]),

        dcc.Tab(label='AI Predictions', children=[
            html.Div([
                html.H2('AI Predictions - Coming Soon!', style={'textAlign': 'center', 'marginTop': '50px'})
            ]),
        ]),
    ]),
])

# Define callback to update graphs based on dropdown selections
@app.callback(
    [Output('energy-output-graph', 'figure'),
     Output('power-consumption-graph', 'figure'),
     Output('battery-levels-graph', 'figure'),
     Output('system-load-voltage-graph', 'figure'),
     Output('inverter-efficiency-gauge', 'figure'),
     Output('system-faults-graph', 'figure'),
     Output('customer-distribution-pie', 'figure'),
     Output('solar-panels-type-pie', 'figure'),
     Output('battery-technology-pie', 'figure')],
    [Input('customer-profile-dropdown', 'value'),
     Input('solar-panels-type-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_graphs(selected_profile, selected_panel_type, selected_month):
    # Filter the data based on selected profile, panel type, and month
    filtered_data = filter_data(selected_profile, selected_panel_type, selected_month)

    # If no data is available for the selection, return empty figures
    if filtered_data.empty:
        return [{} for _ in range(9)]

    # Group by hour
    hourly_trends = group_by_hour(filtered_data)

    # If no data is available after grouping, return empty figures
    if hourly_trends.empty:
        return [{} for _ in range(9)]

    # Create a 'Datetime' column for plotting
    hourly_trends['Datetime'] = pd.to_datetime(hourly_trends['Date'].astype(str) + ' ' + hourly_trends['Hour'].astype(str) + ':00:00')

    # 1. Energy output trend
    fig_energy_output = px.line(hourly_trends, x='Datetime', y='Solar Panels Energy Output (W)', 
                                title='Hourly Solar Panels Energy Output Trend')

    # 2. Power consumption trend
    fig_power_consumption = px.line(hourly_trends, x='Datetime', y='Power Consumption (kW)', 
                                    title='Hourly Power Consumption Trend')

    # 3. Battery energy stored trend
    fig_battery_levels = px.line(hourly_trends, x='Datetime', y='Energy Stored in Batteries (kWh)', 
                                 title='Hourly Battery Levels Trend')

    # 4. System Load & Voltage (Dual Axis)
    fig_system_load_voltage = go.Figure()
    fig_system_load_voltage.add_trace(go.Scatter(x=hourly_trends['Datetime'], y=hourly_trends['System Load (kW)'],
                                                 mode='lines', name='System Load (kW)'))
    fig_system_load_voltage.add_trace(go.Scatter(x=hourly_trends['Datetime'], y=hourly_trends['Voltage (V)'],
                                                 mode='lines', name='Voltage (V)', yaxis='y2'))

    fig_system_load_voltage.update_layout(
        title="System Load and Voltage Trend",
        yaxis=dict(title="System Load (kW)"),
        yaxis2=dict(title="Voltage (V)", overlaying='y', side='right')
    )

    # 5. Inverter Efficiency Gauge
    avg_efficiency = filtered_data['Inverter Efficiency (%)'].mean()
    fig_inverter_efficiency = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_efficiency if pd.notnull(avg_efficiency) else 0,  # Handle NaN values
        title={'text': "Inverter Efficiency (%)"},
        gauge={'axis': {'range': [None, 100]}}
    ))

    # 6. System Faults Over Time (Line Chart)
    fault_data = filtered_data.groupby('Timestamp').sum(numeric_only=True)['System Fault Alerts']
    fig_system_faults = px.line(x=fault_data.index, y=fault_data.values, 
                                 title='System Fault Alerts Over Time',
                                 labels={'x': 'Timestamp', 'y': 'Fault Alerts'})

    # 7. Customer profile distribution (Pie chart)
    customer_dist = solar_data['Customer Profile'].value_counts()
    fig_customer_dist = px.pie(values=customer_dist.values, names=customer_dist.index, 
                               title='Customer Profile Distribution')

    # 8. Solar panel types distribution (Pie chart)
    panel_dist = solar_data['Solar Panels Type'].value_counts()
    fig_panel_dist = px.pie(values=panel_dist.values, names=panel_dist.index, 
                            title='Solar Panels Type Distribution')

    # 9. Battery technology distribution (Pie chart)
    battery_tech_dist = solar_data['Battery Technology'].value_counts()
    fig_battery_tech_dist = px.pie(values=battery_tech_dist.values, names=battery_tech_dist.index, 
                                   title='Battery Technology Distribution')

    return (fig_energy_output, fig_power_consumption, fig_battery_levels,
            fig_system_load_voltage, fig_inverter_efficiency, fig_system_faults,
            fig_customer_dist, fig_panel_dist, fig_battery_tech_dist)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
