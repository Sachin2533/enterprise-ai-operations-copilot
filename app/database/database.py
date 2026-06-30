import sqlite3
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ENTERPRISE_DB = DATA_DIR / "enterprise.db"
MEMORY_DB = DATA_DIR / "memory.db"


def get_enterprise_connection():
    return sqlite3.connect(ENTERPRISE_DB)


def get_memory_connection():
    return sqlite3.connect(MEMORY_DB)


def initialize_memory_database():
    conn = get_memory_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT,
        ai_response TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles(
        user_key TEXT PRIMARY KEY,
        employee_id TEXT
    )
    """)

    conn.commit()
    conn.close()


def initialize_databases():

    # Enterprise DB
    conn = get_enterprise_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees(
        employee_id TEXT PRIMARY KEY,
        name TEXT,
        department TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS leave_balance(
        employee_id TEXT PRIMARY KEY,
        casual_leave INTEGER,
        sick_leave INTEGER
    )
    """)

    cur.execute("""
    INSERT OR IGNORE INTO employees
    VALUES
    ('EMP101','Sachin','AI Team')
    """)

    cur.execute("""
    INSERT OR IGNORE INTO leave_balance
    VALUES
    ('EMP101',8,5)
    """)

    conn.commit()
    conn.close()

    # Memory DB
    initialize_memory_database()


if __name__ == "__main__":
    initialize_databases()
    print("Databases initialized.")
