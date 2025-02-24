import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

class PowerConsumptionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'Temperature (°C)',
            'Solar Panels Energy Output (W)',
            'Energy Stored in Batteries (kWh)',
            'System Load (kW)',
            'Hour',
            'Day',
            'Month',
            'DayOfWeek',
            'IsWeekend'
        ]

    def prepare_data(self, data_path):
        """Prepare data for training"""
        data = pd.read_csv(data_path)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Feature engineering
        data['Hour'] = data['Timestamp'].dt.hour
        data['Day'] = data['Timestamp'].dt.day
        data['Month'] = data['Timestamp'].dt.month
        data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        
        return data

    def train_model(self, data_path):
        """Train the model and save it"""
        # Prepare data
        data = self.prepare_data(data_path)
        
        X = data[self.features]
        y = data['Power Consumption (kW)']

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.features)

        # Split data using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X_scaled))
        train_index, test_index = splits[-1]

        X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Save model artifacts
        self.save_model()
        
        return metrics, X_test, y_test, y_pred

    def calculate_metrics(self, y_true, y_pred):
        """Calculate model performance metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }

    def save_model(self, model_dir='models'):
        """Save the trained model and scaler"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        joblib.dump(self.model, f'{model_dir}/power_consumption_model.joblib')
        joblib.dump(self.scaler, f'{model_dir}/scaler.joblib')

    def load_model(self, model_dir='models'):
        """Load the trained model and scaler"""
        try:
            model_path = f'{model_dir}/power_consumption_model.joblib'
            scaler_path = f'{model_dir}/scaler.joblib'
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print("Warning: Model files not found. Please train the model first.")
                return False
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, input_data):
        """Make predictions on new data"""
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
            
        if not all(feature in input_data.columns for feature in self.features):
            raise ValueError(f"Input data must contain all required features: {self.features}")
            
        scaled_data = self.scaler.transform(input_data[self.features])
        prediction = self.model.predict(scaled_data)
        return prediction

def plot_results(X_test, y_test, y_pred, feature_importance, features):
    """Plot model results and feature importance"""
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(features, feature_importance)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance for Power Consumption Prediction')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Power Consumption')
    plt.xlabel('Time')
    plt.ylabel('Power Consumption (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/prediction_results.png')
    plt.close()

def main():
    # Initialize predictor
    predictor = PowerConsumptionPredictor()
    
    # Train and save modeli
    metrics, X_test, y_test, y_pred = predictor.train_model(
        '/Users/habeeb/Downloads/Git/ML/Powerbox/ETL/Raw_data/Powerbox/Clean_data/cleaned_solar_data.csv'
    )
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Plot results
    feature_importance = predictor.model.feature_importances_
    plot_results(X_test, y_test, y_pred, feature_importance, predictor.features)
    
    print("\nModel and visualizations have been saved in the 'models' directory")

if __name__ == "__main__":
    main()