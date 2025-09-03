import sqlite3
import os
# --------------
# DB setup (lite)
# --------------

# ---------------------------
# Configuration & Directories
# ---------------------------
DATA_ROOT = os.environ.get("BABY_MVP_DATA_ROOT", "./data")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
DB_PATH = os.path.join(DATA_ROOT, "mvp.sqlite")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    audio_path TEXT,
    image_path TEXT,
    is_crying INTEGER,
    cry_score REAL,
    cry_type TEXT,
    emotion TEXT,
    posture TEXT,
    predicted_reason TEXT,
    reason_scores TEXT, -- JSON stringified dict
    caregiver_label TEXT
);
"""

def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

with db_conn() as conn:
    conn.execute(SCHEMA_SQL)
    conn.commit()