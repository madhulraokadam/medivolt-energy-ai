#!/usr/bin/env python3
"""
MediVolt Database Viewer
View contents of the SQLite database
"""

import sqlite3
import os

DATABASE_NAME = 'medivolt.db'

def view_database():
    """View all database contents"""
    
    if not os.path.exists(DATABASE_NAME):
        print(f"Error: Database '{DATABASE_NAME}' not found!")
        print("Run 'python app.py' first to create the database.")
        return
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    print("=" * 60)
    print("       MEDIVOLT DATABASE VIEWER")
    print("=" * 60)
    
    # View tables
    print("\n📋 TABLES:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        print(f"   - {table[0]}")
    
    # View users
    print("\n👥 USERS:")
    cursor.execute('SELECT id, username, email, phone, hospital_id, hospital_name, created_at, last_login, is_active FROM users')
    users = cursor.fetchall()
    if users:
        print(f"   {'ID':<5} {'Username':<12} {'Email':<22} {'Hospital ID':<12} {'Hospital Name':<18} {'Last Login':<18} {'Active'}")
        print("   " + "-" * 105)
        for user in users:
            print(f"   {user[0]:<5} {user[1]:<12} {user[2]:<22} {str(user[4] or ''):<12} {str(user[5] or ''):<18} {str(user[7] or 'Never'):<18} {user[8]}")
    else:
        print("   No users found.")
    
    # View login sessions
    print("\n🔐 LOGIN SESSIONS:")
    cursor.execute('SELECT id, user_id, email, hospital_id, login_time, logout_time, is_active FROM login_sessions ORDER BY login_time DESC LIMIT 10')
    sessions = cursor.fetchall()
    if sessions:
        print(f"   {'ID':<5} {'UserID':<7} {'Email':<22} {'Hospital ID':<12} {'Login Time':<18} {'Logout Time':<18} {'Active'}")
        print("   " + "-" * 95)
        for s in sessions:
            print(f"   {s[0]:<5} {s[1]:<7} {s[2]:<22} {str(s[3] or ''):<12} {str(s[4]):<18} {str(s[5] or 'Active'):<18} {s[6]}")
    else:
        print("   No sessions found.")

    # View activity log
    print("\n📝 ACTIVITY LOG:")
    cursor.execute('SELECT id, user_id, action, details, timestamp FROM activity_log ORDER BY timestamp DESC LIMIT 10')
    activities = cursor.fetchall()
    if activities:
        print(f"   {'ID':<5} {'UserID':<7} {'Action':<15} {'Details':<30} {'Timestamp'}")
        print("   " + "-" * 75)
        for a in activities:
            print(f"   {a[0]:<5} {a[1]:<7} {a[2]:<15} {str(a[3] or '')[:30]:<30} {a[4]}")
    else:
        print("   No activities found.")
    
    # View ML predictions
    print("\n📊 ML PREDICTIONS:")
    cursor.execute('SELECT id, user_id, num_beds, equipment_hours, outdoor_temp, prediction, confidence, created_at FROM ml_predictions ORDER BY created_at DESC LIMIT 20')
    predictions = cursor.fetchall()
    if predictions:
        print(f"   {'ID':<5} {'User':<5} {'Beds':<6} {'EquipHrs':<10} {'Temp':<6} {'Prediction':<12} {'Confidence':<12} {'Created'}")
        print("   " + "-" * 75)
        for pred in predictions:
            print(f"   {pred[0]:<5} {pred[1]:<5} {pred[2]:<6} {pred[3]:<10} {pred[4]:<6} {pred[5]:<12} {pred[6]:<12} {pred[7]}")
    else:
        print("   No predictions found.")
    
    # View ML training history
    print("\n🎓 ML TRAINING HISTORY:")
    cursor.execute('SELECT id, user_id, n_samples, train_mae, test_mae, train_r2, test_r2, created_at FROM ml_training ORDER BY created_at DESC LIMIT 10')
    training = cursor.fetchall()
    if training:
        print(f"   {'ID':<5} {'User':<5} {'Samples':<8} {'Train MAE':<12} {'Test MAE':<12} {'Train R²':<10} {'Test R²':<10} {'Created'}")
        print("   " + "-" * 85)
        for t in training:
            print(f"   {t[0]:<5} {t[1]:<5} {t[2]:<8} {t[3]:<12.2f} {t[4]:<12.2f} {t[5]:<10.4f} {t[6]:<10.4f} {t[7]}")
    else:
        print("   No training history found.")
    
    # Database stats
    print("\n📈 DATABASE STATS:")
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM ml_predictions')
    pred_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM ml_training')
    train_count = cursor.fetchone()[0]
    print(f"   Total Users: {user_count}")
    print(f"   Total Predictions: {pred_count}")
    print(f"   Total Training Sessions: {train_count}")
    
    print("\n" + "=" * 60)
    
    conn.close()


def add_sample_data():
    """Add sample data for testing"""
    import database
    
    # Create sample users
    database.create_user("john_doe", "john@example.com", "password123", "9876543210")
    database.create_user("jane_smith", "jane@example.com", "password456", "1234567890")
    print("Sample users created!")
    
    # Add sample predictions
    sample_features = {
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
    database.save_ml_prediction(1, sample_features, 2500.50, 0.85)
    database.save_ml_prediction(1, sample_features, 2450.25, 0.82)
    print("Sample predictions created!")
    
    # Add sample training
    sample_metrics = {
        'train_mae': 45.2,
        'test_mae': 52.8,
        'train_r2': 0.98,
        'test_r2': 0.95
    }
    database.save_ml_training(1, 1000, sample_metrics)
    print("Sample training history created!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--add-sample':
        add_sample_data()
        print("\nNow view the data:")
    
    view_database()
