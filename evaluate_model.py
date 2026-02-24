#!/usr/bin/env python3
"""
MediVolt ML Model Evaluation
Evaluate model performance with metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
import os

# Generate dataset
def generate_hospital_dataset(n_samples=1000):
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
    
    base_consumption = df['num_beds'] * 15
    equip_impact = df['equipment_hours'] * df['num_beds'] * 2
    hvac_impact = np.where(
        (df['outdoor_temp'] < 15) | (df['outdoor_temp'] > 25),
        df['building_area'] * 0.5,
        df['building_area'] * 0.2
    )
    humidity_impact = df['humidity'] * df['building_area'] * 0.01
    occupancy_impact = df['occupancy_rate'] * df['num_beds'] * 10
    weekend_factor = np.where(df['is_weekend'] == 1, 0.85, 1.0)
    efficiency_impact = (1 - df['hvac_efficiency']) * hvac_impact
    
    df['daily_energy_kwh'] = (
        base_consumption + equip_impact + hvac_impact +
        humidity_impact + occupancy_impact + efficiency_impact
    ) * weekend_factor
    
    df['daily_energy_kwh'] += np.random.normal(0, 50, n_samples)
    df['daily_energy_kwh'] = df['daily_energy_kwh'].clip(lower=100)
    
    return df


def evaluate_model():
    """Evaluate model with multiple metrics and visualizations"""
    
    print("=" * 70)
    print("       MEDIVOLT ML MODEL EVALUATION")
    print("=" * 70)
    
    # Generate data
    print("\n📊 Generating dataset...")
    df = generate_hospital_dataset(2000)  # More samples for better evaluation
    
    feature_names = [
        'num_beds', 'equipment_hours', 'outdoor_temp', 'humidity',
        'day_of_week', 'is_weekend', 'building_area', 'occupancy_rate', 'hvac_efficiency'
    ]
    
    X = df[feature_names].values
    y = df['daily_energy_kwh'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    print("\n" + "=" * 70)
    print("📈 MODEL EVALUATION METRICS")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\n🔹 {name}:")
        
        # Train
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        test_median_ae = median_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        if name == 'Linear Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'y_pred_test': y_pred_test,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'test_median_ae': test_median_ae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │ Training Metrics:                                    │")
        print(f"   │   MAE:  {train_mae:>10.2f} kWh                        │")
        print(f"   │   RMSE: {train_rmse:>10.2f} kWh                        │")
        print(f"   │   R²:   {train_r2:>10.4f}                            │")
        print(f"   ├─────────────────────────────────────────────────────┤")
        print(f"   │ Testing Metrics:                                     │")
        print(f"   │   MAE:  {test_mae:>10.2f} kWh                        │")
        print(f"   │   RMSE: {test_rmse:>10.2f} kWh                        │")
        print(f"   │   R²:   {test_r2:>10.4f}                            │")
        print(f"   │   MAPE: {test_mape:>10.2f} %                          │")
        print(f"   │   Median AE: {test_median_ae:>6.2f} kWh                  │")
        print(f"   ├─────────────────────────────────────────────────────┤")
        print(f"   │ Cross-Validation (5-fold):                         │")
        print(f"   │   Mean R²: {cv_scores.mean():>8.4f} ± {cv_scores.std():.4f}         │")
        print(f"   └─────────────────────────────────────────────────────┘")
    
    # Best model
    best_model = max(results.keys(), key=lambda k: results[k]['test_r2'])
    print(f"\n🏆 Best Model: {best_model} (R² = {results[best_model]['test_r2']:.4f})")
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    create_plots(df, feature_names, X_train, X_test, y_train, y_test, results)
    
    print("\n✅ Evaluation complete! Graphs saved to 'evaluation_plots' folder.")
    
    return results


def create_plots(df, feature_names, X_train, X_test, y_train, y_test, results):
    """Create and save evaluation plots"""
    
    # Create output directory
    output_dir = 'evaluation_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_result = results[best_model_name]
    y_pred_test = best_result['y_pred_test']
    best_model = best_result['model']
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.5, c='blue', edgecolors='k', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Energy Consumption (kWh)', fontsize=12)
    plt.ylabel('Predicted Energy Consumption (kWh)', fontsize=12)
    plt.title(f'Actual vs Predicted Energy Consumption\n{best_model_name} (R² = {best_result["test_r2"]:.4f})', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/actual_vs_predicted.png', dpi=150)
    plt.close()
    print(f"   ✓ Saved: actual_vs_predicted.png")
    
    # 2. Residual Plot
    residuals = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test, residuals, alpha=0.5, c='green', edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Energy Consumption (kWh)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_plot.png', dpi=150)
    plt.close()
    print(f"   ✓ Saved: residual_plot.png")
    
    # 3. Residuals Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Residuals (kWh)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Residuals', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals_distribution.png', dpi=150)
    plt.close()
    print(f"   ✓ Saved: residuals_distribution.png")
    
    # 4. Feature Importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.barh(range(len(importances)), importances[indices], color='teal', edgecolor='black')
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature Importance Ranking', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=150)
        plt.close()
        print(f"   ✓ Saved: feature_importance.png")
    
    # 5. Model Comparison
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    test_r2_scores = [results[m]['test_r2'] for m in model_names]
    test_mae_scores = [results[m]['test_mae'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1 = plt.subplot(1, 2, 1)
    bars1 = ax1.bar(x, test_r2_scores, width, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Comparison - R² Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.set_ylim(0, 1)
    for bar, score in zip(bars1, test_r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{score:.3f}', ha='center', fontsize=10)
    
    ax2 = plt.subplot(1, 2, 2)
    bars2 = ax2.bar(x, test_mae_scores, width, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_ylabel('MAE (kWh)')
    ax2.set_title('Model Comparison - MAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15)
    for bar, score in zip(bars2, test_mae_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                f'{score:.0f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=150)
    plt.close()
    print(f"   ✓ Saved: model_comparison.png")
    
    # 6. Prediction Error Distribution
    plt.figure(figsize=(10, 6))
    error_percent = ((y_test - y_pred_test) / y_test) * 100
    plt.hist(error_percent, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='green', linestyle='--', lw=2)
    plt.xlabel('Prediction Error (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_errors.png', dpi=150)
    plt.close()
    print(f"   ✓ Saved: prediction_errors.png")
    
    # Summary
    print(f"\n📁 All plots saved to: {output_dir}/")
    print(f"   - actual_vs_predicted.png")
    print(f"   - residual_plot.png")
    print(f"   - residuals_distribution.png")
    print(f"   - feature_importance.png")
    print(f"   - model_comparison.png")
    print(f"   - prediction_errors.png")


if __name__ == "__main__":
    evaluate_model()
