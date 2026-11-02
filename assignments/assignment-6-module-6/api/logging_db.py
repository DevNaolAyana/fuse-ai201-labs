import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = os.getenv("LOG_DB_PATH", "logs/predictions.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id     TEXT    NOT NULL UNIQUE,
    timestamp      TEXT    NOT NULL,
    input_features TEXT    NOT NULL,
    fraud_prob     REAL,
    decision       TEXT,
    model_version  TEXT    NOT NULL,
    latency_ms     REAL,
    status_code    INTEGER NOT NULL DEFAULT 200,
    is_simulated   INTEGER NOT NULL DEFAULT 0
);
"""

def connect():
    return sqlite3.connect(DB_PATH, timeout=10)

def init_db():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with closing(connect()) as conn:
        conn.executescript(SCHEMA)

def log_prediction(features, prob, decision, model_version, latency_ms, status_code=200, timestamp=None, is_simulated=0):
    request_id = uuid.uuid4().hex
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    with closing(connect()) as conn, conn:
        conn.execute(
            "INSERT INTO predictions (request_id, timestamp, input_features, fraud_prob, decision, model_version, latency_ms, status_code, is_simulated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, timestamp.isoformat(timespec="seconds"), json.dumps(features),
             prob, decision, model_version, latency_ms, status_code, is_simulated),
        )
    return request_id
