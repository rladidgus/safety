import sqlite3
import json
from datetime import datetime

DB_PATH = "chat_logs.db"

def init_db():
    """Initializes the SQLite database with the logs table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_message TEXT,
            bot_response TEXT,
            intent TEXT,
            safety_score INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_chat(user_msg: str, bot_resp: str, intent: str = "unknown", safety_score: int = None):
    """Logs the chat interaction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_logs (timestamp, user_message, bot_response, intent, safety_score)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), user_msg, bot_resp, intent, safety_score))
    conn.commit()
    conn.close()
