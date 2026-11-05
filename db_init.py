import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    model TEXT,
    timestamp TEXT
)
""")

conn.commit()
conn.close()

print("Database initialized successfully")
