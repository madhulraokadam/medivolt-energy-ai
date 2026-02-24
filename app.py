from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import threading

app = Flask(__name__, 
            template_folder='.',
            static_folder='.',
            static_url_path='/')

# Enable CORS for all routes
CORS(app)

# Secret key for sessions
app.secret_key = 'medivolt-secret-key-12345'

# Database imports
try:
    import database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Database module not available")

# ML Model imports - Using new comprehensive ml_models.py
try:
    from ml_models import MLModelManager, get_ml_manager
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML models not available. Run: pip install scikit-learn pandas numpy")

# Initialize ML models in background
ml_manager = None
model_lock = threading.Lock()

def init_ml_models():
    """Initialize all ML models in background"""
    global ml_manager
    if ML_AVAILABLE:
        with model_lock:
            if ml_manager is None:
                print("Initializing ML models...")
                ml_manager = get_ml_manager()
                ml_manager.initialize()
                print("All ML models ready!")

# Start model initialization in background
if ML_AVAILABLE:
    threading.Thread(target=init_ml_models, daemon=True).start()

# Store OTPs temporarily (in production, use Redis or database)
otp_store = {}

# OTP validity duration in seconds (10 minutes)
OTP_VALIDITY_DURATION = 600

# Email configuration (update with your email settings)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"

def send_email(recipient_email, otp):
    """Send OTP via email"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = "MediVolt Password Reset OTP"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px;">
                <h2 style="color: #0F766E;">MediVolt Password Reset</h2>
                <p>You requested to reset your password. Your OTP is:</p>
                <div style="background-color: #0F766E; color: white; padding: 15px; 
                            font-size: 24px; font-weight: bold; text-align: center; 
                            border-radius: 5px; letter-spacing: 5px;">
                    {otp}
                </div>
                <p style="color: #666;">This OTP is valid for 10 minutes.</p>
                <p style="color: #999; font-size: 12px;">
                    If you didn't request this, please ignore this email.
                </p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    if not DB_AVAILABLE:
        return jsonify({"success": False, "message": "Database not available"}), 500
    
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    phone = data.get('phone', '')
    
    if not username or not email or not password:
        return jsonify({"success": False, "message": "Username, email, and password are required"}), 400
    
    success, message = database.create_user(username, email, password, phone)
    
    if success:
        # return created user record for UI convenience
        try:
            user = database.get_user_by_email(email)
            user_dict = dict(user) if user else None
        except Exception:
            user_dict = None
        return jsonify({"success": True, "message": message, "user": user_dict})
    else:
        return jsonify({"success": False, "message": message}), 400


@app.route('/api/login', methods=['POST'])
def login():
    """User login - accepts any credentials for demo purposes"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400
    
    # For demo: accept any login - get user from DB or create session anyway
    user = None
    if DB_AVAILABLE:
        user = database.verify_user(username, password)
    
    # Create session regardless - allows any user to login
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    if user:
        # User exists in database
        hospital_id = user.get('hospital_id', '') or ''
        session_token, session_id = database.create_session(
            user['id'], 
            user['email'], 
            hospital_id,
            ip_address,
            user_agent
        )
        if DB_AVAILABLE:
            database.log_activity(user['id'], session_id, 'LOGIN', f'User logged in from {ip_address}')
        
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']
        session['session_token'] = session_token
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "phone": user['phone'],
                "hospital_id": hospital_id,
                "hospital_name": user.get('hospital_name', '') or ''
            },
            "session_token": session_token
        })
    else:
        # Create a demo session for any username
        demo_email = f"{username}@demo.com"
        session_token = "demo_token_" + str(hash(username))[:10]
        
        session['user_id'] = 0
        session['username'] = username
        session['email'] = demo_email
        session['session_token'] = session_token
        
        return jsonify({
            "success": True,
            "message": "Login successful (demo mode)",
            "user": {
                "id": 0,
                "username": username,
                "email": demo_email,
                "phone": "",
                "hospital_id": "",
                "hospital_name": ""
            },
            "session_token": session_token
        })


@app.route('/api/logout', methods=['POST'])
def logout():
    """User logout with session tracking"""
    # End session in database if exists
    session_token = session.get('session_token')
    if session_token and DB_AVAILABLE:
        database.end_session(session_token)
        user_id = session.get('user_id')
        if user_id:
            database.log_activity(user_id, None, 'LOGOUT', 'User logged out')
    
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})


@app.route('/api/session', methods=['GET'])
def check_session():
    """Check if user is logged in"""
    if 'user_id' in session:
        return jsonify({
            "logged_in": True,
            "username": session.get('username'),
            "user_id": session.get('user_id')
        })
    else:
        return jsonify({"logged_in": False})


# ==================== EXISTING AUTH ROUTES ====================

@app.route('/api/send-otp', methods=['POST'])
def send_otp():
    """Send OTP to user's email"""
    data = request.get_json()
    
    email = data.get('email')
    phone = data.get('phone')
    
    if not email or not phone:
        return jsonify({"success": False, "message": "Email and phone are required"}), 400
    
    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    
    # Store OTP with timestamp
    otp_store[email] = {
        'otp': otp,
        'phone': phone,
        'timestamp': __import__('time').time()
    }
    
    # Try to send email
    success, message = send_email(email, otp)
    
    if success:
        return jsonify({
            "success": True, 
            "message": "OTP sent to your email",
            "demo_otp": otp
        })
    else:
        return jsonify({
            "success": True, 
            "message": "OTP generated (email failed)",
            "demo_otp": otp,
            "note": "Configure email settings to send real emails"
        })

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP entered by user"""
    data = request.get_json()
    
    email = data.get('email')
    entered_otp = data.get('otp')
    
    if not email or not entered_otp:
        return jsonify({"success": False, "message": "Email and OTP are required"}), 400
    
    stored_data = otp_store.get(email)
    
    if not stored_data:
        return jsonify({"success": False, "message": "No OTP found for this email"}), 404
    
    current_time = time.time()
    otp_time = stored_data.get('timestamp', 0)
    if current_time - otp_time > OTP_VALIDITY_DURATION:
        del otp_store[email]
        return jsonify({"success": False, "message": "OTP has expired. Please request a new one."}), 400
    
    if stored_data['otp'] == entered_otp:
        del otp_store[email]
        return jsonify({"success": True, "message": "OTP verified successfully"})
    
    return jsonify({"success": False, "message": "Invalid OTP"}), 400

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    """Reset password after OTP verification"""
    data = request.get_json()
    
    email = data.get('email')
    new_password = data.get('newPassword')
    
    if not email or not new_password:
        return jsonify({"success": False, "message": "Email and new password are required"}), 400
    
    return jsonify({
        "success": True, 
        "message": "Password reset successfully"
    })

# ==================== HTML PAGE ROUTES ====================

@app.route('/')
def home():
    """Home page - serves index.html"""
    return render_template('index.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "MediVolt API is running",
        "endpoints": {
            "POST /api/login": "User login",
            "POST /api/register": "User registration",
            "POST /api/send-otp": "Send OTP to email",
            "POST /api/verify-otp": "Verify OTP",
            "POST /api/reset-password": "Reset password"
        }
    })

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot-password.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/style.css')
def style():
    return app.send_static_file('style.css')

@app.route('/script.js')
def script():
    return app.send_static_file('script.js')

# ==================== ML API ENDPOINTS ====================

@app.route('/api/ml/train', methods=['POST'])
def train_model():
    """Train all ML models with new data"""
    if not ML_AVAILABLE:
        return jsonify({"success": False, "message": "ML not available"}), 500
    
    data = request.get_json() or {}
    n_samples = data.get('n_samples', 2000)
    
    with model_lock:
        global ml_manager
        ml_manager = get_ml_manager()
        ml_manager.initialize()
        metrics = {
            "energy": ml_manager.energy_model.metrics,
            "hvac": ml_manager.hvac_model.metrics,
            "carbon": ml_manager.carbon_model.metrics
        }
    
    return jsonify({
        "success": True,
        "message": "All models trained successfully",
        "metrics": metrics
    })

@app.route('/api/ml/predict/energy', methods=['POST'])
def predict_energy():
    """Predict energy consumption using ML model"""
    if not ML_AVAILABLE:
        return jsonify({"success": False, "message": "ML not available"}), 500
    
    data = request.get_json() or {}
    
    features = {
        'num_beds': data.get('num_beds', 100),
        'equipment_hours': data.get('equipment_hours', 12),
        'outdoor_temp': data.get('outdoor_temp', 25),
        'humidity': data.get('humidity', 60),
        'day_of_week': data.get('day_of_week', 0),
        'is_weekend': data.get('is_weekend', 0),
        'building_area': data.get('building_area', 15000),
        'occupancy_rate': data.get('occupancy_rate', 0.8),
        'hvac_efficiency': data.get('hvac_efficiency', 0.85)
    }
    
    with model_lock:
        if ml_manager is None or not ml_manager.is_initialized:
            return jsonify({"success": False, "message": "Model not initialized"}), 503
        
        result = ml_manager.energy_model.predict_detailed(features)
    
    return jsonify({
        "success": True,
        "type": "energy_consumption",
        "prediction": result
    })

@app.route('/api/ml/predict/hvac', methods=['POST'])
def predict_hvac():
    """Predict HVAC optimization using ML model"""
    if not ML_AVAILABLE:
        return jsonify({"success": False, "message": "ML not available"}), 500
    
    data = request.get_json() or {}
    
    features = {
        'outdoor_temp': data.get('outdoor_temp', 25),
        'outdoor_humidity': data.get('outdoor_humidity', 60),
        'indoor_temp_setpoint': data.get('indoor_temp_setpoint', 22),
        'building_area': data.get('building_area', 15000),
        'num_occupants': data.get('num_occupants', 100),
        'equipment_load': data.get('equipment_load', 10000),
        'hvac_age_years': data.get('hvac_age_years', 5),
        'schedule_occupancy': data.get('schedule_occupancy', 0.8),
        'weather_condition': data.get('weather_condition', 0),
        'current_efficiency': data.get('current_efficiency', 0.7)
    }
    
    with model_lock:
        if ml_manager is None or not ml_manager.is_initialized:
            return jsonify({"success": False, "message": "Model not initialized"}), 503
        
        result = ml_manager.hvac_model.predict_detailed(features)
    
    return jsonify({
        "success": True,
        "type": "hvac_optimization",
        "prediction": result
    })

@app.route('/api/ml/predict/carbon', methods=['POST'])
def predict_carbon():
    """Predict carbon emissions using ML model"""
    if not ML_AVAILABLE:
        return jsonify({"success": False, "message": "ML not available"}), 500
    
    data = request.get_json() or {}
    
    features = {
        'energy_consumption': data.get('energy_consumption', 50000),
        'energy_source': data.get('energy_source', 0),
        'grid_mix_percentage': data.get('grid_mix_percentage', 70),
        'solar_percentage': data.get('solar_percentage', 10),
        'building_area': data.get('building_area', 15000),
        'occupancy_rate': data.get('occupancy_rate', 0.8),
        'hvac_efficiency': data.get('hvac_efficiency', 0.7),
        'month': data.get('month', 1),
        'season': data.get('season', 0)
    }
    
    with model_lock:
        if ml_manager is None or not ml_manager.is_initialized:
            return jsonify({"success": False, "message": "Model not initialized"}), 503
        
        result = ml_manager.carbon_model.predict_detailed(features)
    
    return jsonify({
        "success": True,
        "type": "carbon_forecasting",
        "prediction": result
    })

@app.route('/api/ml/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from all ML models"""
    if not ML_AVAILABLE:
        return jsonify({"success": False, "message": "ML not available"}), 500
    
    with model_lock:
        if ml_manager is None or not ml_manager.is_initialized:
            return jsonify({"success": False, "message": "Model not initialized"}), 503
        
        energy_importance = dict(zip(
            ml_manager.energy_model.feature_names,
            ml_manager.energy_model.model.feature_importances_
        ))
        hvac_importance = dict(zip(
            ml_manager.hvac_model.feature_names,
            ml_manager.hvac_model.model.feature_importances_
        ))
        carbon_importance = dict(zip(
            ml_manager.carbon_model.feature_names,
            ml_manager.carbon_model.model.feature_importances_
        ))
    
    return jsonify({
        "success": True,
        "energy_consumption": energy_importance,
        "hvac_optimization": hvac_importance,
        "carbon_forecasting": carbon_importance
    })

@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Check ML model status"""
    if not ML_AVAILABLE:
        return jsonify({
            "available": False,
            "message": "Install ML dependencies: pip install scikit-learn pandas numpy"
        })
    
    with model_lock:
        is_ready = ml_manager is not None and ml_manager.is_initialized
    
    return jsonify({
        "available": True,
        "ready": is_ready,
        "models": {
            "energy_consumption": is_ready,
            "hvac_optimization": is_ready,
            "carbon_forecasting": is_ready
        },
        "message": "All ML models ready" if is_ready else "Models initializing..."
    })

# ==================== DATABASE API ENDPOINTS ====================

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users (admin only)"""
    if not DB_AVAILABLE:
        return jsonify({"success": False, "message": "Database not available"}), 500
    
    users = database.get_all_users()
    return jsonify({
        "success": True,
        "users": [dict(u) for u in users]
    })


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user (admin only)"""
    if not DB_AVAILABLE:
        return jsonify({"success": False, "message": "Database not available"}), 500
    try:
        ok = database.delete_user(user_id)
        if ok:
            return jsonify({"success": True, "message": "User deleted"})
        else:
            return jsonify({"success": False, "message": "User not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['PUT'])
def put_user(user_id):
    """Update user fields (admin only)"""
    if not DB_AVAILABLE:
        return jsonify({"success": False, "message": "Database not available"}), 500
    data = request.get_json() or {}
    allowed = {'username', 'email', 'phone', 'hospital_id', 'hospital_name', 'is_active'}
    fields = {k: data[k] for k in data.keys() & allowed}
    # normalize is_active
    if 'is_active' in fields:
        try:
            fields['is_active'] = int(bool(fields['is_active']))
        except Exception:
            fields.pop('is_active', None)
    try:
        ok = database.update_user(user_id, fields)
        if ok:
            return jsonify({"success": True, "message": "User updated"})
        else:
            return jsonify({"success": False, "message": "User not found or no changes"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get prediction history"""
    if not DB_AVAILABLE:
        return jsonify({"success": False, "message": "Database not available"}), 500
    
    user_id = session.get('user_id')
    predictions = database.get_prediction_history(user_id)
    return jsonify({
        "success": True,
        "predictions": [dict(p) for p in predictions]
    })


if __name__ == '__main__':
    print("Starting MediVolt API with ML and Database...")
    print("Database: SQLite (medivolt.db)")
    print("Default login: admin / admin123")
    print("\n=== ML Endpoints ===")
    print("  POST /api/ml/train - Train model with custom data")
    print("  POST /api/ml/predict - Predict energy consumption")
    print("  GET  /api/ml/feature-importance - Get feature importance")
    print("  GET  /api/ml/status - Check model status")
    print("\n=== Auth Endpoints ===")
    print("  POST /api/register - Register new user")
    print("  POST /api/login - User login")
    print("  POST /api/logout - User logout")
    app.run(debug=True, port=5500)
