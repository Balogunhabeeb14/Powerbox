import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ensure the Exploration folder exists
def create_charts_folder(folder='Exploration'):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Plot histograms or KDE plots for numerical columns
def plot_numerical_distributions(df, folder='Exploration'):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'{folder}/{col}_distribution.png')
        plt.close()

# Plot count plots for categorical columns
def plot_categorical_counts(df, folder='Exploration'):
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.savefig(f'{folder}/{col}_countplot.png')
        plt.close()

# Plot correlation heatmap for numerical columns
def plot_correlation_heatmap(df, folder='Exploration'):
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{folder}/correlation_heatmap.png')
    plt.close()

# Scatter plot for pairwise relationships between numerical columns
def plot_pairwise_scatter(df, folder='Exploration'):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    pairplot = sns.pairplot(df[num_columns])
    pairplot.fig.suptitle('Pairwise Scatter Plot', y=1.02)
    pairplot.savefig(f'{folder}/pairwise_scatter.png')
    plt.close()

# Box plot for outlier detection in numerical columns
def plot_box_plots(df, folder='Exploration'):
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=col)
        plt.title(f'Box Plot of {col}')
        plt.savefig(f'{folder}/{col}_boxplot.png')
        plt.close()

# Time series plot for any column vs Timestamp
def plot_time_series(df, timestamp_col, folder='Exploration'):
    if timestamp_col not in df.columns:
        print(f"Timestamp column {timestamp_col} not found.")
        return

    time_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in time_columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df[timestamp_col], df[col])
        plt.title(f'Time Series of {col}')
        plt.xlabel('Timestamp')
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.savefig(f'{folder}/{col}_timeseries.png')
        plt.close()

# Full exploratory script
def explore_data_graphically(file_path, timestamp_col='Timestamp'):
    df = pd.read_csv(file_path)
    
    # Create Exploration folder
    create_charts_folder(folder='Exploration')

    # Plot distributions for numerical columns
    plot_numerical_distributions(df, folder='Exploration')
    
    # Plot count plots for categorical columns
    plot_categorical_counts(df, folder='Exploration')
    
    # Plot correlation heatmap for numerical data
    plot_correlation_heatmap(df, folder='Exploration')
    
    # Plot pairwise scatter plots for numerical columns
    plot_pairwise_scatter(df, folder='Exploration')
    
    # Plot box plots for detecting outliers
    plot_box_plots(df, folder='Exploration')
    
    # Plot time series if timestamp column is present
    if timestamp_col in df.columns:
        plot_time_series(df, timestamp_col, folder='Exploration')

    print("Exploration complete! Check the 'Exploration' folder for the generated plots.")

# usage
file_path = 'solar_energy_data.csv'  
explore_data_graphically(file_path)
