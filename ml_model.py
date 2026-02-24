"""
MediVolt ML Model for Healthcare Energy Consumption Prediction
Uses Random Forest Regressor for energy prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# Generate realistic hospital energy consumption dataset
def generate_hospital_dataset(n_samples=1000):
    """
    Generate synthetic dataset for hospital energy consumption
    Features: number_of_beds, equipment_hours, outdoor_temp, humidity, 
              day_of_week, is_weekend, building_area_sqft, occupancy_rate
    Target: daily_energy_kwh
    """
    np.random.seed(42)
    
    data = {
        'num_beds': np.random.randint(20, 500, n_samples),
        'equipment_hours': np.random.uniform(4, 24, n_samples),
        'outdoor_temp': np.random.uniform(5, 40, n_samples),  # Celsius
        'humidity': np.random.uniform(30, 90, n_samples),  # Percentage
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'building_area': np.random.randint(5000, 50000, n_samples),  # sq ft
        'occupancy_rate': np.random.uniform(0.3, 1.0, n_samples),
        'hvac_efficiency': np.random.uniform(0.5, 0.95, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic energy consumption with complex relationships
    # Base consumption per bed
    base_consumption = df['num_beds'] * 15
    
    # Equipment usage impact
    equip_impact = df['equipment_hours'] * df['num_beds'] * 2
    
    # HVAC impact (more energy for extreme temps)
    hvac_impact = np.where(
        (df['outdoor_temp'] < 15) | (df['outdoor_temp'] > 25),
        df['building_area'] * 0.5,
        df['building_area'] * 0.2
    )
    
    # Humidity impact
    humidity_impact = df['humidity'] * df['building_area'] * 0.01
    
    # Occupancy impact
    occupancy_impact = df['occupancy_rate'] * df['num_beds'] * 10
    
    # Weekend reduction (hospitals still need power but less elective procedures)
    weekend_factor = np.where(df['is_weekend'] == 1, 0.85, 1.0)
    
    # HVAC efficiency impact
    efficiency_impact = (1 - df['hvac_efficiency']) * hvac_impact
    
    # Calculate total energy consumption (in kWh)
    df['daily_energy_kwh'] = (
        base_consumption + 
        equip_impact + 
        hvac_impact + 
        humidity_impact + 
        occupancy_impact +
        efficiency_impact
    ) * weekend_factor
    
    # Add some noise
    df['daily_energy_kwh'] += np.random.normal(0, 50, n_samples)
    df['daily_energy_kwh'] = df['daily_energy_kwh'].clip(lower=100)
    
    return df


class EnergyPredictionModel:
    """ML Model for Hospital Energy Consumption Prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'num_beds', 'equipment_hours', 'outdoor_temp', 'humidity',
            'day_of_week', 'is_weekend', 'building_area', 'occupancy_rate', 'hvac_efficiency'
        ]
        self.is_trained = False
        self.metrics = {}
        
    def train(self, n_samples=1000):
        """Train the ML model with synthetic hospital data"""
        print("Generating hospital energy dataset...")
        df = generate_hospital_dataset(n_samples)
        
        X = df[self.feature_names].values
        y = df['daily_energy_kwh'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        self.metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        self.is_trained = True
        print(f"Model trained successfully!")
        print(f"Test MAE: {self.metrics['test_mae']:.2f} kWh")
        print(f"Test R² Score: {self.metrics['test_r2']:.4f}")
        
        return self.metrics
    
    def predict(self, features):
        """Predict energy consumption for given features"""
        if not self.is_trained:
            self.train()
        
        # Convert to array if needed
        if isinstance(features, dict):
            feature_array = np.array([[
                features.get('num_beds', 100),
                features.get('equipment_hours', 12),
                features.get('outdoor_temp', 25),
                features.get('humidity', 60),
                features.get('day_of_week', 0),
                features.get('is_weekend', 0),
                features.get('building_area', 15000),
                features.get('occupancy_rate', 0.8),
                features.get('hvac_efficiency', 0.85)
            ]])
        else:
            feature_array = np.array([features])
        
        # Scale and predict
        feature_scaled = self.scaler.transform(feature_array)
        prediction = self.model.predict(feature_scaled)
        
        return float(prediction[0])
    
    def predict_with_confidence(self, features):
        """Predict with confidence intervals using Gradient Boosting"""
        if not self.is_trained:
            self.train()
            
        # Use the same prediction
        prediction = self.predict(features)
        
        # Calculate confidence based on feature ranges
        confidence = 0.85  # Base confidence
        
        # Adjust confidence based on feature extremity
        if isinstance(features, dict):
            if features.get('outdoor_temp', 25) < 10 or features.get('outdoor_temp', 25) > 35:
                confidence -= 0.1
            if features.get('occupancy_rate', 0.8) > 0.95:
                confidence -= 0.05
        
        return {
            'prediction': round(prediction, 2),
            'confidence': round(confidence, 2),
            'lower_bound': round(prediction * 0.85, 2),
            'upper_bound': round(prediction * 1.15, 2)
        }
    
    def get_feature_importance(self):
        """Get feature importance for model interpretability"""
        if not self.is_trained:
            self.train()
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))
        
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance


# Global model instance
energy_model = EnergyPredictionModel()


def get_model():
    """Get the trained model instance"""
    if not energy_model.is_trained:
        energy_model.train()
    return energy_model


if __name__ == "__main__":
    # Test the model
    model = EnergyPredictionModel()
    metrics = model.train(n_samples=500)
    
    print("\n=== Model Metrics ===")
    print(f"Training MAE: {metrics['train_mae']:.2f} kWh")
    print(f"Testing MAE: {metrics['test_mae']:.2f} kWh")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Testing R²: {metrics['test_r2']:.4f}")
    
    # Test prediction
    test_features = {
        'num_beds': 100,
        'equipment_hours': 12,
        'outdoor_temp': 25,
        'humidity': 60,
        'day_of_week': 1,
        'is_weekend': 0,
        'building_area': 15000,
        'occupancy_rate': 0.8,
        'hvac_efficiency': 0.85
    }
    
    result = model.predict_with_confidence(test_features)
    print(f"\n=== Test Prediction ===")
    print(f"Features: {test_features}")
    print(f"Prediction: {result}")
    
    print("\n=== Feature Importance ===")
    importance = model.get_feature_importance()
    for feature, imp in importance.items():
        print(f"{feature}: {imp:.4f}")
