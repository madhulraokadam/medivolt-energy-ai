"""
MediVolt ML Models - Complete ML Suite
Includes:
1. Energy Consumption Prediction
2. HVAC Optimization
3. Carbon Emissions Forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import threading
import os

# ==================== DATASET GENERATORS ====================

def generate_energy_dataset(n_samples=2000):
    """Generate healthcare energy consumption dataset"""
    np.random.seed(42)
    
    data = {
        'num_beds': np.random.randint(20, 500, n_samples),
        'equipment_hours': np.random.uniform(4, 24, n_samples),
        'outdoor_temp': np.random.uniform(5, 40, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'building_area': np.random.randint(5000, 50000, n_samples),
        'occupancy_rate': np.random.uniform(0.3, 1.0, n_samples),
        'hvac_efficiency': np.random.uniform(0.5, 0.95, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    base = df['num_beds'] * 15
    equip = df['equipment_hours'] * df['num_beds'] * 2
    hvac_impact = np.where((df['outdoor_temp'] < 15) | (df['outdoor_temp'] > 25), df['building_area'] * 0.5, df['building_area'] * 0.2)
    humidity_impact = df['humidity'] * df['building_area'] * 0.01
    occupancy_impact = df['occupancy_rate'] * df['num_beds'] * 10
    weekend_factor = np.where(df['is_weekend'] == 1, 0.85, 1.0)
    efficiency_impact = (1 - df['hvac_efficiency']) * hvac_impact
    
    df['daily_energy_kwh'] = (base + equip + hvac_impact + humidity_impact + occupancy_impact + efficiency_impact) * weekend_factor
    df['daily_energy_kwh'] += np.random.normal(0, 50, n_samples)
    df['daily_energy_kwh'] = df['daily_energy_kwh'].clip(lower=100)
    
    return df


def generate_hvac_dataset(n_samples=2000):
    """Generate HVAC optimization dataset"""
    np.random.seed(43)
    
    data = {
        'outdoor_temp': np.random.uniform(5, 42, n_samples),
        'outdoor_humidity': np.random.uniform(20, 95, n_samples),
        'indoor_temp_setpoint': np.random.uniform(18, 26, n_samples),
        'building_area': np.random.randint(5000, 50000, n_samples),
        'num_occupants': np.random.randint(10, 500, n_samples),
        'equipment_load': np.random.uniform(1000, 50000, n_samples),
        'hvac_age_years': np.random.uniform(1, 20, n_samples),
        'schedule_occupancy': np.random.uniform(0.3, 1.0, n_samples),
        'weather_condition': np.random.randint(0, 4, n_samples),  # 0=sunny, 1=cloudy, 2=rainy, 3=humid
        'current_efficiency': np.random.uniform(0.4, 0.95, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate optimal efficiency based on conditions
    temp_diff = abs(df['outdoor_temp'] - df['indoor_temp_setpoint'])
    age_factor = 1 - (df['hvac_age_years'] * 0.02)
    load_factor = 1 + (df['equipment_load'] / df['building_area']) * 0.01
    occupancy_factor = 1 + (df['num_occupants'] / df['building_area']) * 10
    
    df['optimal_efficiency'] = (
        (0.5 + 0.3 * age_factor + 0.2 * (1 - temp_diff / 30)) * 
        np.clip(load_factor * occupancy_factor, 0.5, 1.5)
    )
    df['optimal_efficiency'] = df['optimal_efficiency'].clip(0.3, 0.98)
    
    # Energy savings potential
    df['energy_savings_potential'] = ((df['optimal_efficiency'] - df['current_efficiency']) / df['current_efficiency']) * 100
    df['energy_savings_potential'] = df['energy_savings_potential'].clip(-20, 40)
    
    # Optimal setpoint
    df['optimal_setpoint'] = df['indoor_temp_setpoint'] + np.where(
        df['outdoor_temp'] > 25, 
        -0.3 * (df['outdoor_temp'] - 25),
        0.2 * (15 - df['outdoor_temp'])
    )
    df['optimal_setpoint'] = df['optimal_setpoint'].clip(16, 28)
    
    return df


def generate_carbon_dataset(n_samples=2000):
    """Generate carbon emissions forecasting dataset"""
    np.random.seed(44)
    
    data = {
        'energy_consumption': np.random.uniform(10000, 100000, n_samples),
        'energy_source': np.random.choice([0, 1, 2, 3], n_samples),  # 0=grid, 1=solar, 2=gas, 3=mixed
        'grid_mix_percentage': np.random.uniform(0, 100, n_samples),
        'solar_percentage': np.random.uniform(0, 50, n_samples),
        'building_area': np.random.randint(5000, 50000, n_samples),
        'occupancy_rate': np.random.uniform(0.3, 1.0, n_samples),
        'hvac_efficiency': np.random.uniform(0.5, 0.95, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'season': np.random.randint(0, 4, n_samples),  # 0=spring, 1=summer, 2=autumn, 3=winter
    }
    
    df = pd.DataFrame(data)
    
    # Emission factors (kg CO2 per kWh)
    emission_factors = {
        'grid': 0.42,
        'solar': 0.05,
        'gas': 0.50,
        'mixed': 0.35
    }
    
    # Calculate carbon emissions
    grid_emissions = (df['energy_consumption'] * df['grid_mix_percentage'] / 100) * emission_factors['grid']
    solar_emissions = (df['energy_consumption'] * df['solar_percentage'] / 100) * emission_factors['solar']
    remaining = 100 - df['grid_mix_percentage'] - df['solar_percentage']
    remaining = remaining.clip(0, 100)
    
    # Seasonal adjustment
    seasonal_factor = np.where(df['season'] == 1, 1.3,  # summer - high AC
                     np.where(df['season'] == 3, 1.2,  # winter - heating
                     np.where(df['season'] == 0, 0.9, 0.95)))  # spring/autumn
    
    # Efficiency adjustment
    efficiency_factor = 1 + (1 - df['hvac_efficiency']) * 0.3
    
    df['carbon_emissions_kg'] = (grid_emissions + solar_emissions + remaining/100 * df['energy_consumption'] * emission_factors['mixed']) * seasonal_factor * efficiency_factor
    df['carbon_emissions_kg'] = df['carbon_emissions_kg'].clip(100, 50000)
    
    # Carbon intensity
    df['carbon_intensity'] = df['carbon_emissions_kg'] / (df['energy_consumption'] / 1000)  # kg CO2 per MWh
    
    # Target reduction (what if we increased solar by 10%?)
    df['potential_reduction'] = df['energy_consumption'] * 0.1 * emission_factors['grid'] * seasonal_factor / 100
    
    return df


# ==================== ML MODEL CLASSES ====================

class EnergyPredictionModel:
    """Energy Consumption Prediction Model"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = ['num_beds', 'equipment_hours', 'outdoor_temp', 'humidity', 'day_of_week', 'is_weekend', 'building_area', 'occupancy_rate', 'hvac_efficiency']
        self.is_trained = False
        self.metrics = {}
        
    def train(self, n_samples=2000):
        df = generate_energy_dataset(n_samples)
        X = df[self.feature_names].values
        y = df['daily_energy_kwh'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        self.is_trained = True
        return self.metrics
    
    def predict(self, features):
        if not self.is_trained:
            self.train()
        feature_array = np.array([[features.get(f, 0) for f in self.feature_names]])
        return float(self.model.predict(feature_array)[0])
    
    def predict_detailed(self, features):
        prediction = self.predict(features)
        return {
            'prediction': round(prediction, 2),
            'unit': 'kWh',
            'confidence': 0.95,
            'lower_bound': round(prediction * 0.85, 2),
            'upper_bound': round(prediction * 1.15, 2)
        }


class HVACOptimizationModel:
    """HVAC Optimization Model"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.feature_names = ['outdoor_temp', 'outdoor_humidity', 'indoor_temp_setpoint', 'building_area', 'num_occupants', 'equipment_load', 'hvac_age_years', 'schedule_occupancy', 'weather_condition', 'current_efficiency']
        self.is_trained = False
        self.metrics = {}
        
    def train(self, n_samples=2000):
        df = generate_hvac_dataset(n_samples)
        X = df[self.feature_names].values
        y = df['optimal_efficiency'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        self.is_trained = True
        return self.metrics
    
    def predict(self, features):
        if not self.is_trained:
            self.train()
        feature_array = np.array([[features.get(f, 0) for f in self.feature_names]])
        return float(self.model.predict(feature_array)[0])
    
    def predict_detailed(self, features):
        current_eff = features.get('current_efficiency', 0.7)
        optimal_eff = self.predict(features)
        
        savings = ((optimal_eff - current_eff) / current_eff) * 100
        
        return {
            'optimal_efficiency': round(optimal_eff, 4),
            'current_efficiency': current_eff,
            'efficiency_improvement': round(optimal_eff - current_eff, 4),
            'energy_savings_percent': round(max(0, savings), 2),
            'optimal_setpoint': round(21 + (features.get('outdoor_temp', 25) - 20) * 0.1, 1),
            'recommendations': self._generate_recommendations(features, optimal_eff)
        }
    
    def _generate_recommendations(self, features, optimal_eff):
        recs = []
        if features.get('hvac_age_years', 5) > 10:
            recs.append("Consider replacing HVAC system older than 10 years")
        if features.get('current_efficiency', 0.7) < optimal_eff - 0.1:
            recs.append("Schedule maintenance to improve efficiency")
        if features.get('schedule_occupancy', 0.5) < 0.5:
            recs.append("Implement smart scheduling for off-peak hours")
        if not recs:
            recs.append("HVAC system is operating optimally")
        return recs


class CarbonForecastingModel:
    """Carbon Emissions Forecasting Model"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.feature_names = ['energy_consumption', 'energy_source', 'grid_mix_percentage', 'solar_percentage', 'building_area', 'occupancy_rate', 'hvac_efficiency', 'month', 'season']
        self.is_trained = False
        self.metrics = {}
        
    def train(self, n_samples=2000):
        df = generate_carbon_dataset(n_samples)
        X = df[self.feature_names].values
        y = df['carbon_emissions_kg'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        self.is_trained = True
        return self.metrics
    
    def predict(self, features):
        if not self.is_trained:
            self.train()
        feature_array = np.array([[features.get(f, 0) for f in self.feature_names]])
        return float(self.model.predict(feature_array)[0])
    
    def predict_detailed(self, features):
        current_emissions = self.predict(features)
        energy_kwh = features.get('energy_consumption', 50000)
        solar_pct = features.get('solar_percentage', 10)
        
        # Calculate reduction potential
        max_solar_reduction = energy_kwh * 0.2 * 0.37 * 0.9  # 20% more solar, 0.37 kg/kWh, 90% achievable
        efficiency_improvement = (1 - features.get('hvac_efficiency', 0.7)) * energy_kwh * 0.1
        
        return {
            'current_emissions': round(current_emissions, 2),
            'unit': 'kg CO2',
            'carbon_intensity': round(current_emissions / (energy_kwh / 1000), 2),
            'potential_reduction_solar': round(max_solar_reduction, 2),
            'potential_reduction_efficiency': round(efficiency_improvement, 2),
            'total_reduction_potential': round(max_solar_reduction + efficiency_improvement, 2),
            'recommendations': self._generate_recommendations(features, current_emissions)
        }
    
    def _generate_recommendations(self, features, emissions):
        recs = []
        if features.get('solar_percentage', 10) < 20:
            recs.append("Increase solar panel installation to reduce grid dependency")
        if features.get('hvac_efficiency', 0.7) < 0.8:
            recs.append("Improve HVAC efficiency to lower emissions")
        if features.get('energy_source', 0) == 0:
            recs.append("Consider transitioning to renewable energy sources")
        if not recs:
            recs.append("Carbon footprint is well optimized")
        return recs


# ==================== MODEL MANAGER ====================

class MLModelManager:
    """Central manager for all ML models"""
    
    def __init__(self):
        self.energy_model = EnergyPredictionModel()
        self.hvac_model = HVACOptimizationModel()
        self.carbon_model = CarbonForecastingModel()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize all models"""
        print("\n" + "="*60)
        print("Initializing ML Models...")
        print("="*60)
        
        print("\n1️⃣ Training Energy Consumption Model...")
        e_metrics = self.energy_model.train(2000)
        print(f"   R²: {e_metrics['r2']:.4f} | MAE: {e_metrics['mae']:.2f} kWh | MAPE: {e_metrics['mape']:.2f}%")
        
        print("\n2️⃣ Training HVAC Optimization Model...")
        h_metrics = self.hvac_model.train(2000)
        print(f"   R²: {h_metrics['r2']:.4f} | MAE: {h_metrics['mae']:.4f} | MAPE: {h_metrics['mape']:.2f}%")
        
        print("\n3️⃣ Training Carbon Forecasting Model...")
        c_metrics = self.carbon_model.train(2000)
        print(f"   R²: {c_metrics['r2']:.4f} | MAE: {c_metrics['mae']:.2f} kg | MAPE: {c_metrics['mape']:.2f}%")
        
        self.is_initialized = True
        print("\n" + "="*60)
        print("✅ All ML Models Ready!")
        print("="*60)


# Global instance
ml_manager = MLModelManager()

def get_ml_manager():
    return ml_manager
