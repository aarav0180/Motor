"""
tests/test_server_database.py
-----------------------------
Unit tests for the Tier 2 SQLite time-series storage and CRUD functions.
"""

import time
import pytest
from server.database import (
    init_db,
    insert_telemetry,
    get_latest_telemetry,
    get_telemetry_history,
    insert_alert,
    get_alerts,
    acknowledge_alert,
    insert_feedback,
    get_feedback_history,
    get_config,
    set_config,
)


@pytest.fixture
def test_db(tmp_path):
    """Provides an isolated test database path."""
    db_file = str(tmp_path / "test_server.db")
    init_db(db_file)
    return db_file


class TestDatabaseCRUD:
    def test_init_db_creates_tables(self, test_db):
        import sqlite3
        conn = sqlite3.connect(test_db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        assert "telemetry" in tables
        assert "alerts" in tables
        assert "human_feedback" in tables
        assert "system_config" in tables
        conn.close()

    def test_telemetry_insert_and_query(self, test_db):
        t1 = time.time() - 10
        t2 = time.time()

        insert_telemetry(
            {"timestamp": t1, "score": 0.002, "label": "normal", "detector": "autoencoder"},
            db_path=test_db,
        )
        insert_telemetry(
            {
                "timestamp": t2,
                "score": 0.612,
                "label": "anomaly",
                "detector": "hybrid_edge",
                "diagnosis": "inner_race_fault",
                "confidence": 0.998,
                "proc_time_s": 0.002,
            },
            db_path=test_db,
        )

        latest = get_latest_telemetry(db_path=test_db)
        assert latest is not None
        assert latest["score"] == 0.612
        assert latest["diagnosis"] == "inner_race_fault"
        assert latest["confidence"] == 0.998

        history = get_telemetry_history(limit=10, db_path=test_db)
        assert len(history) == 2
        assert history[0]["timestamp"] == t1
        assert history[1]["timestamp"] == t2

    def test_alerts_insert_and_acknowledge(self, test_db):
        a1_id = insert_alert(
            {"message": "BEARING DEFECT", "score": 0.85, "diagnosis": "outer_race_fault"},
            db_path=test_db,
        )
        assert a1_id > 0

        unacked = get_alerts(unacknowledged_only=True, db_path=test_db)
        assert len(unacked) == 1
        assert unacked[0]["acknowledged"] == 0

        # Acknowledge alert
        success = acknowledge_alert(a1_id, db_path=test_db)
        assert success is True

        unacked_after = get_alerts(unacknowledged_only=True, db_path=test_db)
        assert len(unacked_after) == 0

        all_alerts = get_alerts(unacknowledged_only=False, db_path=test_db)
        assert len(all_alerts) == 1
        assert all_alerts[0]["acknowledged"] == 1
        assert all_alerts[0]["ack_timestamp"] is not None

    def test_human_feedback_recording(self, test_db):
        insert_feedback(
            {
                "event": "FALSE_ALARM",
                "score": 0.048,
                "adapted_threshold": 0.0504,
                "notes": "Testing line disturbance",
            },
            db_path=test_db,
        )

        fb_history = get_feedback_history(limit=10, db_path=test_db)
        assert len(fb_history) == 1
        assert fb_history[0]["event"] == "FALSE_ALARM"
        assert fb_history[0]["adapted_threshold"] == 0.0504

    def test_system_config_set_and_get(self, test_db):
        set_config("alert_threshold", 0.075, db_path=test_db)
        val = get_config("alert_threshold", db_path=test_db)
        assert float(val) == 0.075

        # Update existing key
        set_config("alert_threshold", 0.082, db_path=test_db)
        val_updated = get_config("alert_threshold", db_path=test_db)
        assert float(val_updated) == 0.082
