import os
import pandas as pd
import sqlite3
import datetime
import hashlib
import shutil

#Calculates the hash value of files
def md5_hash(file_path):
    """
    Calculate the MD5 hash of a file.
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

#Step O : Check if file exists 
def check_if_file_processed(directory_path, file_path):
   
    # Get the MD5 hash of the target file
    target_hash = md5_hash(file_path)
    if target_hash is None:
        return False

    # Iterate over all files in the directory
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_in_dir = os.path.join(root, file_name)
            # Calculate MD5 hash for each file in the directory
            file_hash = md5_hash(file_in_dir)
            if file_hash == target_hash:
                return True  # Return True as soon as a match is found

    return False  # No match found


def ingest_data(file_path, archive_dir):
    if (file_path.endswith('.csv')) | (file_path.endswith('.xlsx')):
        directory = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        today = datetime.datetime.now().strftime("%Y%m%d")
        file_name_date = file_name.split('.')[0] + "_" + today + '.' + file_name.split('.')[1]
        
        # Check if the directory exists before changing
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        os.chdir(directory)
        os.rename(file_name, file_name_date)

        filecheck = check_if_file_processed(archive_dir, file_name_date)
        if filecheck:
            raise ValueError(f"File {file_name} already processed.")

        if file_name_date.endswith('.csv'):
            df = pd.read_csv(file_name_date)
        elif file_name_date.endswith('.xlsx'):
            df = pd.read_excel(file_name_date)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    return df, file_name_date
    
# Step 2: Validate the file - schema check
def validate_dataframe_columns(dataframe, csv_file_path):

    # Load the CSV file and get the expected column names
    expected_columns = pd.read_csv(csv_file_path).columns.tolist()
    
    # Get the columns of the DataFrame
    actual_columns = dataframe.columns.tolist()
    
    # Check for mismatched columns
    missing_in_dataframe = set(expected_columns) - set(actual_columns)
    extra_in_dataframe = set(actual_columns) - set(expected_columns)
    
    if missing_in_dataframe or extra_in_dataframe:
        error_message = "Column mismatch detected:\n"
        if missing_in_dataframe:
            error_message += f"Missing in DataFrame: {missing_in_dataframe}\n"
        if extra_in_dataframe:
            error_message += f"Extra in DataFrame: {extra_in_dataframe}\n"
        raise ValueError(error_message)
    else:
        print("Columns are valid and match the expected structure.")

# Step 2.1: Correct Data Types
def correct_data_types(df):
    """
    Correct the data types for the dataset to ensure consistency.
    """
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)  # Convert to datetime
        df['System ON'] = df['System ON'].astype(bool)
        df['Temperature (°C)'] = df['Temperature (°C)'].astype(float)
        df['Solar Panels Energy Output (W)'] = df['Solar Panels Energy Output (W)'].astype(float)
        df['Power Consumption (kW)'] = df['Power Consumption (kW)'].astype(float)
        df['Energy Stored in Batteries (kWh)'] = df['Energy Stored in Batteries (kWh)'].astype(float)
        df['Inverter Efficiency (%)'] = df['Inverter Efficiency (%)'].astype(float)
        df['System Load (kW)'] = df['System Load (kW)'].astype(float)
        df['System Fault Alerts'] = df['System Fault Alerts'].astype(bool)
        df['Voltage (V)'] = df['Voltage (V)'].astype(float)
        df['Current (A)'] = df['Current (A)'].astype(float)
        df['Power Factor'] = df['Power Factor'].astype(float)
        df['Dust and Dirt Accumulation (g/m²)'] = df['Dust and Dirt Accumulation (g/m²)'].astype(float)
        df['Battery Low Flag'] = df['Battery Low Flag'].astype(bool)
        df['Battery Full Flag'] = df['Battery Full Flag'].astype(bool)
        df['Customer Profile'] = df['Customer Profile'].astype('category')
        df['User Coordinates'] = df['User Coordinates'].astype(str)  # Keep as string for later processing
        df['Solar Panels Type'] = df['Solar Panels Type'].astype('category')
        df['Solar Panels Configuration'] = df['Solar Panels Configuration'].astype('category')
        df['Depth of Discharge'] = df['Depth of Discharge'].str.rstrip('%').astype(float) / 100  # Convert percentage to float
        df['Battery Capacity (Wh)'] = df['Battery Capacity (Wh)'].astype(float)
        df['Inverter Capacity (kW)'] = df['Inverter Capacity (kW)'].astype(float)
        df['Battery Technology'] = df['Battery Technology'].astype('category')
        
        # Split coordinates into Latitude and Longitude
        df[['Latitude', 'Longitude']] = df['User Coordinates'].str.split(',', expand=True).astype(float)
        
        # Drop the original 'User Coordinates' column
        df.drop(columns=['User Coordinates'], inplace=True)
        
        print("Data types corrected successfully.")
        return df
    
    except Exception as e:
        raise ValueError(f"Error correcting data types: {e}")

# Step 3: Drop columns with missingness > 50%
def drop_high_missingness(df, threshold=0.5):
    missing_percent = df.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > threshold].index
    df = df.drop(columns=columns_to_drop)
    return df
    
# Step 4: Remove outliers using Interquartile Range (IQR)
def remove_outliers(df):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Step 5: Check for data inconsistencies
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

# Step 6: Fill remaining missing values
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])  # Fill categorical with mode
    return df

# Step 7: Load Data into SQLite and Save to CSV
def load_and_save_data(df, db_name, table_name, csv_name, folder):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Load Data into SQLite
    db_path = os.path.join(folder, db_name)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

    # Save cleaned data to CSV
    csv_path = os.path.join(folder, csv_name)
    df.to_csv(csv_path, index=False)

    print(f"Data successfully saved to SQLite database at {db_path}")
    print(f"Cleaned data CSV saved at {csv_path}")
    
# Step 8: Archive the input file
def archive_file(file_path, archive_dir):
   
    try:
        # Ensure the file exists
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return None

        # Create the archive directory if it doesn't exist
        os.makedirs(archive_dir, exist_ok=True)

        # Construct the destination path
        file_name = os.path.basename(file_path)
        archive_path = os.path.join(archive_dir, file_name)

        # Move the file to the archive directory
        shutil.move(file_path, archive_path)

        print(f"File archived to: {archive_path}")
        
    except Exception as e:
        print(f"Failed to archive file: {e}")
        return None

# Full Pipeline Function
def data_pipeline(file_path, db_name, table_name, csv_name, schema_file, archive_dir,folder):

    # Step 1: Ingest data
    df,file_path_date = ingest_data(file_path,archive_dir)
    
    # Step 2: Validate the file 
    validate_dataframe_columns(df, schema_file)

    # Step 2.1: Correct data types
    df = correct_data_types(df)
    
    # Step 3: Drop columns with high missingness
    df = drop_high_missingness(df)
    
    # Step 4: Remove outliers
    df = remove_outliers(df)
    
    # Step 5: Check for inconsistencies
    df = check_inconsistencies(df)
    
    # Step 6: Fill remaining missing values
    df = fill_missing_values(df)
    
    # Step 7: Load data into SQLite and save CSV
    load_and_save_data(df, db_name, table_name, csv_name, folder)
    
    # Step 8: Archive the input file
    archive_file(file_path_date,archive_dir)
    
    print("Data pipeline completed successfully!")

  
# Main Execution Block
if __name__ == "__main__":
    # Define parameters
    file_path = 'ETL/Raw_data/powerbox_dataset_prototype_20241205.csv'  # Path to raw input data
    db_name = 'solar_system.db'                               # Database name
    table_name = 'cleaned_solar_data'                         # Table name in SQLite
    csv_name = 'cleaned_solar_data.csv'                       # Name for cleaned data CSV
    folder = './Powerbox/Clean_data/'                         # Folder to save cleaned data
    schema_file = 'powerbox_schema.csv'                       # Schema file for validation
    archive_dir = './Powerbox/archive/'                       # Archive folder for input files

    # Run the pipeline
    data_pipeline(file_path, db_name, table_name, csv_name, schema_file, archive_dir, folder)