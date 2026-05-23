import sqlite3

conn = sqlite3.connect('workspace/test_jobs.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# View a table
cursor.execute("SELECT * FROM jobs")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()