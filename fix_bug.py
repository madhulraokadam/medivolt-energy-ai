# Fix for sqlite3.Row bug in app.py

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix: Convert sqlite3.Row to dict before using .get()
old_code = '''    user = database.verify_user(username, password)
    
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
        })'''

new_code = '''    user = database.verify_user(username, password)
    
    if user:
        # Convert sqlite3.Row to dict
        user_dict = dict(user)
        
        # Create session in database for tracking
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        hospital_id = user_dict.get('hospital_id', '') or ''
        
        session_token, session_id = database.create_session(
            user_dict['id'], 
            user_dict['email'], 
            hospital_id,
            ip_address,
            user_agent
        )
        
        # Log the login activity
        database.log_activity(user_dict['id'], session_id, 'LOGIN', f'User logged in from {ip_address}')
        
        # Store session info
        session['user_id'] = user_dict['id']
        session['username'] = user_dict['username']
        session['email'] = user_dict['email']
        session['session_token'] = session_token
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user_dict['id'],
                "username": user_dict['username'],
                "email": user_dict['email'],
                "phone": user_dict.get('phone', ''),
                "hospital_id": hospital_id,
                "hospital_name": user_dict.get('hospital_name', '') or ''
            },
            "session_token": session_token
        })'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("sqlite3.Row bug fixed!")
else:
    print("Code not found - may already be fixed")

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
