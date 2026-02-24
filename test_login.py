import sqlite3
import hashlib

conn = sqlite3.connect('medivolt.db')
cursor = conn.cursor()
cursor.execute('SELECT password_hash FROM users WHERE username="admin"')
stored = cursor.fetchone()
if stored:
    print('Stored:', stored[0])
    computed = hashlib.sha256('admin123'.encode()).hexdigest()
    print('Computed:', computed)
    print('Match:', stored[0] == computed)
conn.close()
