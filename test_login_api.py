import requests
import json

# Test login with admin
data = {"username": "admin", "password": "admin123"}
response = requests.post("http://127.0.0.1:5500/api/login", json=data)
print("Status:", response.status_code)
print("Response:", response.json())
