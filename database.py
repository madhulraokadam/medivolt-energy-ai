"""
MediVolt Database Module
Using SQLite for storing user login details and ML data
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DATABASE_NAME = 'medivolt.db'

def init_db():
    """Initialize database with tables"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Users table for login
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            phone TEXT,
            hospital_id TEXT,
            hospital_name TEXT,
            hospital_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # Real-time login sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_token TEXT UNIQUE,
            email TEXT,
            hospital_id TEXT,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            logout_time TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Real-time activity log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (session_id) REFERENCES login_sessions (id)
        )
    ''')
    
    # ML Predictions history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            num_beds INTEGER,
            equipment_hours REAL,
            outdoor_temp REAL,
            humidity REAL,
            day_of_week INTEGER,
            is_weekend INTEGER,
            building_area INTEGER,
            occupancy_rate REAL,
            hvac_efficiency REAL,
            prediction REAL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # ML Training history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_training (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            n_samples INTEGER,
            train_mae REAL,
            test_mae REAL,
            train_r2 REAL,
            test_r2 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert default admin user (admin/admin123)
    default_password = hashlib.sha256('admin123'.encode()).hexdigest()
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, phone)
            VALUES (?, ?, ?, ?)
        ''', ('admin', 'admin@medivolt.com', default_password, '1234567890'))
    except sqlite3.IntegrityError:
        pass  # Admin already exists
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {DATABASE_NAME}")


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, email, password, phone=''):
    """Create a new user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, phone)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, phone))
        conn.commit()
        return True, "User created successfully"
    except sqlite3.IntegrityError as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()


def verify_user(username, password):
    """Verify user login credentials"""
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = hash_password(password)
    
    cursor.execute('''
        SELECT id, username, email, phone, created_at, last_login, is_active, hospital_id, hospital_name
        FROM users 
        WHERE username = ? AND password_hash = ?
    ''', (username, password_hash))
    
    user = cursor.fetchone()
    
    if user:
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now(), user['id']))
        conn.commit()
    
    conn.close()
    return user


def get_user_by_id(user_id):
    """Get user by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_all_users():
    """Get all users"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, phone, created_at, last_login, is_active FROM users')
    users = cursor.fetchall()
    conn.close()
    return users


def save_ml_prediction(user_id, features, prediction, confidence):
    """Save ML prediction to database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ml_predictions (
            user_id, num_beds, equipment_hours, outdoor_temp, humidity,
            day_of_week, is_weekend, building_area, occupancy_rate,
            hvac_efficiency, prediction, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, features.get('num_beds'), features.get('equipment_hours'),
        features.get('outdoor_temp'), features.get('humidity'),
        features.get('day_of_week'), features.get('is_weekend'),
        features.get('building_area'), features.get('occupancy_rate'),
        features.get('hvac_efficiency'), prediction, confidence
    ))
    conn.commit()
    conn.close()


def get_prediction_history(user_id=None, limit=10):
    """Get ML prediction history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute('''
            SELECT * FROM ml_predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT * FROM ml_predictions 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
    predictions = cursor.fetchall()
    conn.close()
    return predictions


def save_ml_training(user_id, n_samples, metrics):
    """Save ML training history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ml_training (
            user_id, n_samples, train_mae, test_mae, train_r2, test_r2
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        user_id, n_samples, metrics.get('train_mae'), metrics.get('test_mae'),
        metrics.get('train_r2'), metrics.get('test_r2')
    ))
    conn.commit()
    conn.close()


def get_training_history(user_id=None, limit=10):
    """Get ML training history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute('''
            SELECT * FROM ml_training 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT * FROM ml_training 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
    training = cursor.fetchall()
    conn.close()
    return training


# Initialize database on import
init_db()


# ===== NEW FUNCTIONS FOR REAL-TIME TRACKING =====

def update_user_hospital_info(user_id, hospital_id, hospital_name, hospital_address):
    """Update user with hospital information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users 
        SET hospital_id = ?, hospital_name = ?, hospital_address = ?
        WHERE id = ?
    ''', (hospital_id, hospital_name, hospital_address, user_id))
    conn.commit()
    conn.close()


def create_session(user_id, email, hospital_id, ip_address='', user_agent=''):
    """Create a new login session"""
    import uuid
    conn = get_db_connection()
    cursor = conn.cursor()
    session_token = str(uuid.uuid4())
    
    cursor.execute('''
        INSERT INTO login_sessions (user_id, session_token, email, hospital_id, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, session_token, email, hospital_id, ip_address, user_agent))
    
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return session_token, session_id


def end_session(session_token):
    """End a login session"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE login_sessions 
        SET logout_time = ?, is_active = 0 
        WHERE session_token = ?
    ''', (datetime.now(), session_token))
    conn.commit()
    conn.close()


def get_active_sessions(user_id=None):
    """Get active login sessions"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute('''
            SELECT * FROM login_sessions 
            WHERE user_id = ? AND is_active = 1 
            ORDER BY login_time DESC
        ''', (user_id,))
    else:
        cursor.execute('''
            SELECT * FROM login_sessions 
            WHERE is_active = 1 
            ORDER BY login_time DESC
        ''')
    sessions = cursor.fetchall()
    conn.close()
    return sessions


def log_activity(user_id, session_id, action, details):
    """Log user activity"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO activity_log (user_id, session_id, action, details)
        VALUES (?, ?, ?, ?)
    ''', (user_id, session_id, action, details))
    conn.commit()
    conn.close()


def get_activity_log(user_id=None, limit=50):
    """Get activity log"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute('''
            SELECT * FROM activity_log 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT * FROM activity_log 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
    activities = cursor.fetchall()
    conn.close()
    return activities


def get_user_by_email(email):
    """Get user by email"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_hospital_id(hospital_id):
    """Get user by hospital ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE hospital_id = ?', (hospital_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def delete_user(user_id):
    """Delete a user by id"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def update_user(user_id, fields: dict):
    """Update user fields. fields is a dict of column->value"""
    if not fields:
        return False
    conn = get_db_connection()
    cursor = conn.cursor()
    set_clauses = []
    values = []
    for k, v in fields.items():
        set_clauses.append(f"{k} = ?")
        values.append(v)
    values.append(user_id)
    sql = f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?"
    cursor.execute(sql, tuple(values))
    conn.commit()
    ok = cursor.rowcount > 0
    conn.close()
    return ok


if __name__ == "__main__":
    # Test database
    print("Testing database...")
    
    # Create test user
    success, msg = create_user("testuser", "test@example.com", "test123", "9876543210")
    print(f"Create user: {msg}")
    
    # Verify user
    user = verify_user("admin", "admin123")
    print(f"Admin login: {'Success' if user else 'Failed'}")
    
    # Get all users
    users = get_all_users()
    print(f"Total users: {len(users)}")
    for u in users:
        print(f"  - {u['username']} ({u['email']})")
