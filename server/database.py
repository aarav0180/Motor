"""
server/database.py — SQLite time-series storage with WAL mode.

Provides thread-safe persistence for:
  - `telemetry`: periodic 10-second vibration health updates
  - `alerts`: threshold breach events
  - `human_feedback`: technician fault confirmations & false alarm adaptations
  - `system_config`: active dynamic thresholds and settings
"""

import contextlib
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .config import settings

logger = logging.getLogger("server.database")


def _get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return an SQLite connection configured with WAL mode."""
    target_path = Path(db_path or settings.database_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(target_path),
        timeout=15.0,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    # High-throughput concurrent settings
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


@contextlib.contextmanager
def get_db(db_path: Optional[str] = None):
    """Context manager yielding a transactional SQLite connection."""
    conn = _get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and initial configuration."""
    with get_db(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                score REAL NOT NULL,
                label TEXT NOT NULL,
                detector TEXT NOT NULL,
                diagnosis TEXT DEFAULT 'healthy',
                confidence REAL DEFAULT 1.0,
                proc_time_s REAL DEFAULT 0.0
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry(timestamp);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                message TEXT NOT NULL,
                score REAL NOT NULL,
                diagnosis TEXT DEFAULT 'anomaly',
                acknowledged INTEGER DEFAULT 0,
                ack_timestamp REAL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event TEXT NOT NULL,
                score REAL NOT NULL,
                adapted_threshold REAL,
                notes TEXT
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT UNIQUE NOT NULL,
                p256dh TEXT NOT NULL,
                auth TEXT NOT NULL,
                client_id TEXT,
                created_at REAL NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_push_endpoint ON push_subscriptions(endpoint);")

        # Seed initial alert threshold if not set
        cur = conn.cursor()
        cur.execute("SELECT value FROM system_config WHERE key = 'alert_threshold'")
        if not cur.fetchone():
            now = time.time()
            cur.execute(
                "INSERT INTO system_config (key, value, updated_at) VALUES (?, ?, ?)",
                ("alert_threshold", str(settings.default_threshold), now),
            )
    logger.info("Initialized database schema at %s", db_path or settings.database_path)


# ---------------------------------------------------------------------------
# Telemetry CRUD
# ---------------------------------------------------------------------------

def insert_telemetry(record: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Insert a telemetry snapshot received from Edge."""
    ts = float(record.get("ts") or record.get("timestamp") or time.time())
    score = float(record.get("score", 0.0))
    label = str(record.get("label", "normal"))
    detector = str(record.get("detector", "autoencoder"))
    diagnosis = str(record.get("diagnosis", "healthy"))
    confidence = float(record.get("confidence", 1.0))
    proc_time_s = float(record.get("proc_time_s", 0.0))

    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO telemetry
                (timestamp, score, label, detector, diagnosis, confidence, proc_time_s)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, score, label, detector, diagnosis, confidence, proc_time_s),
        )
        return int(cur.lastrowid)


def get_latest_telemetry(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the single most recent telemetry record."""
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, score, label, detector, diagnosis, confidence, proc_time_s
            FROM telemetry
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_telemetry_history(
    limit: int = 60,
    start: Optional[float] = None,
    end: Optional[float] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query telemetry history with optional time window."""
    query = "SELECT id, timestamp, score, label, detector, diagnosis, confidence, proc_time_s FROM telemetry"
    params: List[Any] = []
    conditions = []

    if start is not None:
        conditions.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        conditions.append("timestamp <= ?")
        params.append(end)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        # Return in ascending chronological order for charts
        return [dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Alerts CRUD
# ---------------------------------------------------------------------------

def insert_alert(alert: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Insert an alert event triggered by Edge."""
    ts = float(alert.get("ts") or alert.get("timestamp") or time.time())
    message = str(alert.get("message", "FAULT DETECTED"))
    score = float(alert.get("score", 0.0))
    diagnosis = str(alert.get("diagnosis", "anomaly"))

    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alerts (timestamp, message, score, diagnosis, acknowledged)
            VALUES (?, ?, ?, ?, 0)
            """,
            (ts, message, score, diagnosis),
        )
        return int(cur.lastrowid)


def get_alerts(
    unacknowledged_only: bool = False,
    limit: int = 20,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query recent alerts."""
    query = "SELECT id, timestamp, message, score, diagnosis, acknowledged, ack_timestamp FROM alerts"
    params: List[Any] = []

    if unacknowledged_only:
        query += " WHERE acknowledged = 0"

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


def acknowledge_alert(alert_id: int, db_path: Optional[str] = None) -> bool:
    """Mark an alert as acknowledged by technician."""
    now = time.time()
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE alerts SET acknowledged = 1, ack_timestamp = ? WHERE id = ?",
            (now, alert_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Human Feedback CRUD
# ---------------------------------------------------------------------------

def insert_feedback(feedback: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Record human feedback and adapted threshold."""
    ts = float(feedback.get("timestamp") or time.time())
    event = str(feedback.get("event", "FALSE_ALARM"))
    score = float(feedback.get("score", 0.0))
    adapted = feedback.get("adapted_threshold")
    notes = feedback.get("notes")

    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO human_feedback (timestamp, event, score, adapted_threshold, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, event, score, adapted, notes),
        )
        return int(cur.lastrowid)


def get_feedback_history(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, event, score, adapted_threshold, notes
            FROM human_feedback
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# System Configuration CRUD
# ---------------------------------------------------------------------------

def get_config(key: str, default: Any = None, db_path: Optional[str] = None) -> Any:
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM system_config WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default


def set_config(key: str, value: Any, db_path: Optional[str] = None) -> None:
    now = time.time()
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO system_config (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, str(value), now),
        )


# ---------------------------------------------------------------------------
# Push Subscriptions CRUD (Web Push / Disconnected App Pub/Sub)
# ---------------------------------------------------------------------------

def add_push_subscription(
    endpoint: str,
    p256dh: str,
    auth: str,
    client_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Store or update a client device push notification subscription."""
    now = time.time()
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO push_subscriptions (endpoint, p256dh, auth, client_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
                p256dh = excluded.p256dh,
                auth = excluded.auth,
                client_id = excluded.client_id,
                created_at = excluded.created_at
            """,
            (endpoint, p256dh, auth, client_id, now),
        )
        return int(cur.lastrowid or 0)


def remove_push_subscription(endpoint: str, db_path: Optional[str] = None) -> bool:
    """Remove an expired or unsubscribed push registration."""
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
        return cur.rowcount > 0


def get_all_push_subscriptions(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all active push subscriptions for alert broadcasting."""
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, endpoint, p256dh, auth, client_id, created_at FROM push_subscriptions")
        return [dict(r) for r in cur.fetchall()]


def count_push_subscriptions(db_path: Optional[str] = None) -> int:
    """Return count of registered devices."""
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM push_subscriptions")
        row = cur.fetchone()
        return int(row["count"]) if row else 0
