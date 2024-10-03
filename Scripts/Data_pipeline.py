import os
import pandas as pd
import sqlite3

# Step 1: Ingest Data
def ingest_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    return df

# Step 2: Drop columns with missingness > 50%
def drop_high_missingness(df, threshold=0.5):
    missing_percent = df.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > threshold].index
    df = df.drop(columns=columns_to_drop)
    return df

# Step 3: Remove outliers using Interquartile Range (IQR)
def remove_outliers(df):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Step 4: Check for data inconsistencies
def check_inconsistencies(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Remove invalid values (e.g., negative values for energy-related columns)
    energy_columns = ['Solar Panels Energy Output (W)', 'Power Consumption (kW)',
                      'Energy Stored in Batteries (kWh)', 'System Load (kW)', 
                      'Battery Capacity (Wh)', 'Inverter Capacity (kW)']
    
    for column in energy_columns:
        if column in df.columns:
            df = df[df[column] >= 0]  # Ensure no negative values for these columns

    return df

# Step 5: Fill remaining missing values
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])  # Fill categorical with mode
    return df

# Step 6: Load Data into SQLite and Save to CSV
def load_and_save_data(df, db_name, table_name, csv_name, folder):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Load Data into SQLite
    db_path = os.path.join(folder, db_name)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

    # Save cleaned data to CSV
    csv_path = os.path.join(folder, csv_name)
    df.to_csv(csv_path, index=False)

    print(f"Data successfully saved to SQLite database at {db_path}")
    print(f"Cleaned data CSV saved at {csv_path}")

# Full Pipeline Function
def data_pipeline(file_path, db_name, table_name, csv_name, folder):
    # Step 1: Ingest data
    df = ingest_data(file_path)
    
    # Step 2: Drop columns with high missingness
    df = drop_high_missingness(df)
    
    # Step 3: Remove outliers
    df = remove_outliers(df)
    
    # Step 4: Check for inconsistencies
    df = check_inconsistencies(df)
    
    # Step 5: Fill remaining missing values
    df = fill_missing_values(df)
    
    # Step 6: Load data into SQLite and save CSV
    load_and_save_data(df, db_name, table_name, csv_name, folder)
    
    print("Data pipeline completed successfully!")

# usage
file_path = 'solar_energy_data.csv'  # Or 'solar_energy_data.xlsx'
db_name = 'solar_system.db'
table_name = 'cleaned_solar_data'
csv_name = 'cleaned_solar_data.csv'
folder = 'Cleaned_data'

data_pipeline(file_path, db_name, table_name, csv_name, folder)
