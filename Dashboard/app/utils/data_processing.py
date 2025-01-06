import pandas as pd

def filter_data(data, profile, panel_type, month):
    """
    Filter the data based on profile, panel type, and month.
    """
    filtered = data[(data['Customer Profile'] == profile) & (data['Solar Panels Type'] == panel_type)]
    if month:
        filtered = filtered[filtered['Timestamp'].dt.month == month]
    return filtered

def group_by_hour(data):
    """
    Group the filtered data by hour for analysis.
    """
    data['Hour'] = data['Timestamp'].dt.hour
    data['Date'] = data['Timestamp'].dt.date
    return data.groupby(['Date', 'Hour']).mean(numeric_only=True).reset_index()
