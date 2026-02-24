# Fix script for app.py
import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the login function
old_code = '''@app.route('/api/login', methods=['POST'])
def login():
    """User login with database verification"""
    if not DB_AVAILABLE:
        return jsonify({"success": False, "message": "Database not available"}), 500
    
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400
    
    user = database.verify_user(username, password)
    
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "phone": user['phone']
            }
        })
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401'''

new_code = '''@app.route('/api/login', methods=['POST'])
def login():
    """User login with database verification and real-time tracking"""
    if not DB_AVAILABLE:
        return jsonify({"success": False, "message": "Database not available"}), 500
    
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400
    
    user = database.verify_user(username, password)
    
    if user:
        # Create session in database for tracking
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        hospital_id = user.get('hospital_id', '') or ''
        
        session_token, session_id = database.create_session(
            user['id'], 
            user['email'], 
            hospital_id,
            ip_address,
            user_agent
        )
        
        # Log the login activity
        database.log_activity(user['id'], session_id, 'LOGIN', f'User logged in from {ip_address}')
        
        # Store session info
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
        return jsonify({"success": False, "message": "Invalid credentials"}), 401'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("Login function updated!")
else:
    print("Login function not found - may already be updated")

# Now fix logout function
old_logout = '''@app.route('/api/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})'''

new_logout = '''@app.route('/api/logout', methods=['POST'])
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
    return jsonify({"success": True, "message": "Logged out successfully"})'''

if old_logout in content:
    content = content.replace(old_logout, new_logout)
    print("Logout function updated!")
else:
    print("Logout function not found - may already be updated")

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("All fixes applied!")
