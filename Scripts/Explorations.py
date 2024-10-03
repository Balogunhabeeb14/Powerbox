import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define constants for file paths
BASE_PATH = '/Users/habeeb/Downloads/Git/ML/Powerbox'
CLEANED_DATA_FILE = os.path.join(BASE_PATH, 'Clean_data/cleaned_solar_data.csv')
EXPLORATION_FOLDER = os.path.join(BASE_PATH, 'Exploration')

# Ensure the Exploration folder exists
def create_charts_folder(folder=EXPLORATION_FOLDER):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to sanitize file names by replacing unsupported characters
def sanitize_filename(name):
    return name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("°", "").replace("²", "")

# Plot histograms or KDE plots for numerical columns
def plot_numerical_distributions(df, folder=EXPLORATION_FOLDER):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        sanitized_col = sanitize_filename(col)
        plt.figure(figsize=(8, 6))
        try:
            if df[col].nunique() > 1:  # Only plot KDE if there is variance
                sns.kdeplot(df[col].dropna(), shade=True)
                plt.title(f'Distribution of {col} (KDE)')
                plt.savefig(f'{folder}/{sanitized_col}_kde_distribution.png')
                plt.close()

            # Additionally plot histograms using matplotlib
            plt.figure(figsize=(8, 6))
            plt.hist(df[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
            plt.title(f'Histogram of {col}')
            plt.savefig(f'{folder}/{sanitized_col}_histogram.png')
            plt.close()
        except Exception as e:
            print(f"Could not plot {col} due to: {e}")

# Plot count plots for categorical columns
def plot_categorical_counts(df, folder=EXPLORATION_FOLDER):
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_columns:
        sanitized_col = sanitize_filename(col)
        plt.figure(figsize=(8, 6))
        try:
            sns.countplot(data=df, x=col)
            plt.title(f'Count Plot of {col}')
            plt.xticks(rotation=45)
            plt.savefig(f'{folder}/{sanitized_col}_countplot.png')
            plt.close()
        except Exception as e:
            print(f"Could not plot {col} due to: {e}")

# Plot correlation heatmap for numerical columns
def plot_correlation_heatmap(df, folder=EXPLORATION_FOLDER):
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{folder}/correlation_heatmap.png')
    plt.close()

# Scatter plot for pairwise relationships between numerical columns
def plot_pairwise_scatter(df, folder=EXPLORATION_FOLDER):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    pairplot = sns.pairplot(df[num_columns])
    pairplot.fig.suptitle('Pairwise Scatter Plot', y=1.02)
    pairplot.savefig(f'{folder}/pairwise_scatter.png')
    plt.close()

# Box plot for outlier detection in numerical columns
def plot_box_plots(df, folder=EXPLORATION_FOLDER):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        sanitized_col = sanitize_filename(col)
        plt.figure(figsize=(8, 6))
        try:
            sns.boxplot(data=df, x=col)
            plt.title(f'Box Plot of {col}')
            plt.savefig(f'{folder}/{sanitized_col}_boxplot.png')
            plt.close()
        except Exception as e:
            print(f"Could not plot {col} due to: {e}")

# Time series plot for any column vs Timestamp
def plot_time_series(df, timestamp_col, folder=EXPLORATION_FOLDER):
    if timestamp_col not in df.columns:
        print(f"Timestamp column {timestamp_col} not found.")
        return

    time_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in time_columns:
        sanitized_col = sanitize_filename(col)
        plt.figure(figsize=(10, 6))
        try:
            plt.plot(df[timestamp_col], df[col])
            plt.title(f'Time Series of {col}')
            plt.xlabel('Timestamp')
            plt.ylabel(col)
            plt.xticks(rotation=45)
            plt.savefig(f'{folder}/{sanitized_col}_timeseries.png')
            plt.close()
        except Exception as e:
            print(f"Could not plot {col} due to: {e}")

# Full exploratory script
def explore_data_graphically(file_path=CLEANED_DATA_FILE, timestamp_col='Timestamp'):
    df = pd.read_csv(file_path)
    
    # Create Exploration folder
    create_charts_folder(folder=EXPLORATION_FOLDER)

    # Plot distributions for numerical columns
    plot_numerical_distributions(df, folder=EXPLORATION_FOLDER)
    
    # Plot count plots for categorical columns
    plot_categorical_counts(df, folder=EXPLORATION_FOLDER)
    
    # Plot correlation heatmap for numerical data
    plot_correlation_heatmap(df, folder=EXPLORATION_FOLDER)
    
    # Plot pairwise scatter plots for numerical columns
    plot_pairwise_scatter(df, folder=EXPLORATION_FOLDER)
    
    # Plot box plots for detecting outliers
    plot_box_plots(df, folder=EXPLORATION_FOLDER)
    
    # Plot time series if timestamp column is present
    if timestamp_col in df.columns:
        plot_time_series(df, timestamp_col, folder=EXPLORATION_FOLDER)

    print("Exploration complete! Check the 'Exploration' folder for the generated plots.")

# Usage
explore_data_graphically()
