#!/usr/bin/env python3
"""
MediVolt ML Dataset Viewer
View and export the healthcare energy consumption dataset
"""

import numpy as np
import pandas as pd
from ml_model import generate_hospital_dataset

def view_dataset(n_samples=20):
    """View the ML dataset"""
    
    print("=" * 70)
    print("       MEDIVOLT ML DATASET - Healthcare Energy Consumption")
    print("=" * 70)
    
    # Generate dataset
    df = generate_hospital_dataset(n_samples=1000)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Columns:")
    for col in df.columns:
        print(f"   - {col}")
    
    print("\n📈 First 20 Records:")
    print(df.head(20).to_string())
    
    print("\n📊 Dataset Statistics:")
    print(df.describe().to_string())
    
    print("\n🔍 Data Types:")
    print(df.dtypes.to_string())
    
    return df


def export_dataset(filename='medivolt_dataset.csv', n_samples=1000):
    """Export dataset to CSV"""
    df = generate_hospital_dataset(n_samples=n_samples)
    df.to_csv(filename, index=False)
    print(f"✅ Dataset exported to {filename}")
    return df


def view_detailed_sample():
    """View detailed sample with explanations"""
    
    print("\n" + "=" * 70)
    print("       DATASET FEATURE EXPLANATIONS")
    print("=" * 70)
    
    explanations = {
        'num_beds': 'Number of hospital beds (20-500)',
        'equipment_hours': 'Daily equipment usage hours (4-24)',
        'outdoor_temp': 'Outdoor temperature in Celsius (5-40)',
        'humidity': 'Humidity percentage (30-90%)',
        'day_of_week': 'Day of week (0=Monday, 6=Sunday)',
        'is_weekend': 'Is weekend (0=No, 1=Yes)',
        'building_area': 'Building area in sq ft (5000-50000)',
        'occupancy_rate': 'Bed occupancy rate (0.3-1.0)',
        'hvac_efficiency': 'HVAC system efficiency (0.5-0.95)',
        'daily_energy_kwh': 'Daily energy consumption in kWh (TARGET)'
    }
    
    print("\n📖 Feature Descriptions:")
    for feature, desc in explanations.items():
        print(f"   {feature:<20} : {desc}")
    
    # Show correlation with target
    df = generate_hospital_dataset(1000)
    print("\n📊 Correlation with Energy Consumption:")
    correlations = df.corr()['daily_energy_kwh'].drop('daily_energy_kwh').sort_values(ascending=False)
    for feature, corr in correlations.items():
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"   {feature:<20} : {sign}{abs(corr):.3f} {bar}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--export':
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            export_dataset(n_samples=n)
        elif sys.argv[1] == '--explain':
            view_detailed_sample()
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python view_dataset.py           - View dataset sample")
            print("  python view_dataset.py --export - Export full dataset to CSV")
            print("  python view_dataset.py --explain - View feature explanations")
        else:
            print("Unknown option. Use --help")
    else:
        view_dataset()
        view_detailed_sample()
