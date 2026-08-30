"""
Database manager for the HR dashboard.

Handles persistence of check-in records, computed DSI scores, and
alert notifications using SQLite.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


class DatabaseManager:
    """
    SQLite-based persistence layer for the HR dashboard.

    Attributes
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "hr_dashboard.db") -> None:
        """
        Initialize the database manager.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the necessary tables if they do not exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tbs REAL NOT NULL,
                vbs REAL NOT NULL,
                abs REAL NOT NULL,
                dsi REAL NOT NULL,
                severity TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                dsi REAL NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0
            )
            """
        )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Check-in operations
    # ------------------------------------------------------------------

    def save_checkin(
        self,
        employee_id: str,
        tbs: float,
        vbs: float,
        abs_: float,
        dsi: float,
        severity: str,
    ) -> int:
        """
        Save a check-in record.

        Parameters
        ----------
        employee_id : str
            Employee identifier.
        tbs : float
            Text-based score.
        vbs : float
            Vision-based score.
        abs_ : float
            Audio-based score.
        dsi : float
            Fused depression severity index.
        severity : str
            Severity level name.

        Returns
        -------
        int
            The ID of the inserted record.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO checkins
                (employee_id, timestamp, tbs, vbs, abs, dsi, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (employee_id, timestamp, tbs, vbs, abs_, dsi, severity),
        )

        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        return record_id

    def get_checkins(
        self,
        employee_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, object]]:
        """
        Retrieve check-in records.

        Parameters
        ----------
        employee_id : str, optional
            Filter by employee ID.
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list of dict
            List of check-in records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if employee_id:
            cursor.execute(
                """
                SELECT * FROM checkins
                WHERE employee_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (employee_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM checkins
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Alert operations
    # ------------------------------------------------------------------

    def save_alert(
        self,
        employee_id: str,
        dsi: float,
        severity: str,
        message: str,
    ) -> int:
        """
        Save an alert notification.

        Parameters
        ----------
        employee_id : str
            Employee identifier.
        dsi : float
            Fused DSI score.
        severity : str
            Severity level name.
        message : str
            Alert message.

        Returns
        -------
        int
            The ID of the inserted alert.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO alerts
                (employee_id, timestamp, dsi, severity, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (employee_id, timestamp, dsi, severity, message),
        )

        conn.commit()
        alert_id = cursor.lastrowid
        conn.close()

        return alert_id

    def get_alerts(
        self,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, object]]:
        """
        Retrieve alert records.

        Parameters
        ----------
        acknowledged : bool, optional
            Filter by acknowledged status.
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list of dict
            List of alert records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if acknowledged is not None:
            cursor.execute(
                """
                SELECT * FROM alerts
                WHERE acknowledged = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(acknowledged), limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def acknowledge_alert(self, alert_id: int) -> None:
        """
        Mark an alert as acknowledged.

        Parameters
        ----------
        alert_id : int
            ID of the alert to acknowledge.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_summary_stats(self) -> Dict[str, object]:
        """
        Compute summary statistics across all check-ins.

        Returns
        -------
        dict
            Dictionary with total check-ins, average DSI, and
            severity distribution.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM checkins")
        total_checkins = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(dsi) FROM checkins")
        avg_dsi = cursor.fetchone()[0] or 0.0

        cursor.execute(
            """
            SELECT severity, COUNT(*) as count
            FROM checkins
            GROUP BY severity
            """
        )
        severity_distribution = {
            row["severity"]: row["count"] for row in cursor.fetchall()
        }

        cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
        pending_alerts = cursor.fetchone()[0]

        conn.close()

        return {
            "total_checkins": total_checkins,
            "avg_dsi": round(float(avg_dsi), 2),
            "severity_distribution": severity_distribution,
            "pending_alerts": pending_alerts,
        }