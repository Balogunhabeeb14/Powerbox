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


# Step 1: Ingest Data
def ingest_data(file_path,archive_dir):
    
    if (file_path.endswith('.csv')) | (file_path.endswith('.xlsx')) :
        #Rename the file by adding the date
        directory = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        today=datetime.datetime.now().strftime("%Y%m%d")
        file_name_date= file_name.split('.')[0]+"_"+today+'.'+file_name.split('.')[1]

        os.chdir(directory)
        os.rename(file_name,file_name_date)
        
        #Check if the file is alredy processed 
        filecheck = check_if_file_processed(archive_dir, file_name_date)
        
        if filecheck == True:
            raise ValueError("File {} already processed.".format(file_name))
    
           
        if file_name_date.endswith('.csv'):
            df = pd.read_csv(file_name_date)
        elif file_name_date.endswith('.xlsx'):
            df = pd.read_excel(file_name_date)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    return df,file_name_date
    
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

  
file_path = './powerbox_dataset_prototype.csv'  # Or 'solar_energy_data.xlsx'
db_name = 'solar_system.db'
table_name = 'cleaned_solar_data'
csv_name = 'cleaned_solar_data.csv'
folder = './Powerbox/Clean_data/'
schema_file = 'powerbox_schema.csv'
archive_dir = './Powerbox/archive/'

data_pipeline(file_path, db_name, table_name, csv_name,schema_file,archive_dir, folder)
