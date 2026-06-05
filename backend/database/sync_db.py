"""
backend/database/sync_db.py

Synchronous SQLite helper for writing events and updating slot states 
from the background detector thread.
"""

import sqlite3
from datetime import datetime
import os

DB_PATH = "backend/parking.db"

def get_connection():
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_sync_db():
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                slot_id TEXT,
                plate TEXT,
                ocr_conf REAL,
                event_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dwell_secs INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS slot_states (
                slot_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'free',
                track_id INTEGER,
                plate TEXT,
                entry_time DATETIME,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def log_event(track_id: int, slot_id: str, plate: Optional[str], ocr_conf: Optional[float], event_type: str, dwell_secs: Optional[int] = None):
    conn = get_connection()
    try:
        now_str = datetime.utcnow().isoformat()
        conn.execute(
            """
            INSERT INTO parking_events (track_id, slot_id, plate, ocr_conf, event_type, timestamp, dwell_secs)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (track_id, slot_id, plate, ocr_conf, event_type, now_str, dwell_secs)
        )
        conn.commit()
    except Exception as e:
        print(f"[Sync DB Error] Failed to log event: {e}")
    finally:
        conn.close()

def update_slot_state(slot_id: str, status: str, track_id: Optional[int] = None, plate: Optional[str] = None, entry_time: Optional[datetime] = None):
    conn = get_connection()
    try:
        now_str = datetime.utcnow().isoformat()
        entry_time_str = entry_time.isoformat() if entry_time else None
        
        # Check if slot state already exists
        row = conn.execute("SELECT slot_id FROM slot_states WHERE slot_id = ?", (slot_id,)).fetchone()
        
        if row is None:
            conn.execute(
                """
                INSERT INTO slot_states (slot_id, status, track_id, plate, entry_time, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (slot_id, status, track_id, plate, entry_time_str, now_str)
            )
        else:
            conn.execute(
                """
                UPDATE slot_states 
                SET status = ?, track_id = ?, plate = ?, entry_time = ?, updated_at = ?
                WHERE slot_id = ?
                """,
                (status, track_id, plate, entry_time_str, now_str, slot_id)
            )
        conn.commit()
    except Exception as e:
        print(f"[Sync DB Error] Failed to update slot state for slot {slot_id}: {e}")
    finally:
        conn.close()
