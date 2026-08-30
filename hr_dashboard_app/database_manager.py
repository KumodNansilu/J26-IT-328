"""
Database manager for the HR dashboard.

Handles persistence of check-in records, computed DSI scores, and
alert notifications using SQLite.

Tables:
    - hr_users : HR personnel login credentials
    - checkins : Check-in session logs (TBS, VBS, ABS, DSI, Delta, Risk Tier, Masking Alert)
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core_fusion_engine.config import DB_PATH


class DatabaseManager:
    """
    SQLite-based persistence layer for the HR dashboard.

    Attributes
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        """
        Initialize the database manager.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite database file.
        """
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the necessary tables and pre-seed HR login."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # HR users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS hr_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'HR',
                created_at TEXT
            )
            """
        )

        # Check-ins table
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
                delta REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                masking_alert TEXT
            )
            """
        )

        # Pre-seed default HR login
        cursor.execute(
            "SELECT COUNT(*) FROM hr_users WHERE username = 'admin@company.com'"
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                """
                INSERT INTO hr_users (username, password, full_name, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "admin@company.com",
                    "admin123",
                    "System Administrator",
                    "HR Admin",
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # HR User operations
    # ------------------------------------------------------------------

    def validate_login(self, username: str, password: str) -> Optional[Dict[str, object]]:
        """
        Validate HR user credentials.

        Parameters
        ----------
        username : str
            HR user email/username.
        password : str
            HR user password.

        Returns
        -------
        dict or None
            User record if credentials are valid, None otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM hr_users
            WHERE username = ? AND password = ?
            """,
            (username, password),
        )
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

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
        delta: float,
        risk_tier: str,
        masking_alert: Optional[str] = None,
    ) -> int:
        """
        Save a check-in record.

        Parameters
        ----------
        employee_id : str
            Employee identifier.
        tbs : float
            Text-based score in [0, 1].
        vbs : float
            Vision-based score in [0, 1].
        abs_ : float
            Audio-based score in [0, 1].
        dsi : float
            Fused Depression Severity Index in [0, 1].
        delta : float
            Discordance Delta.
        risk_tier : str
            Risk tier: "GREEN", "YELLOW", or "RED".
        masking_alert : str, optional
            Masking alert message if |Δ| > 0.50.

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
                (employee_id, timestamp, tbs, vbs, abs, dsi, delta, risk_tier, masking_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (employee_id, timestamp, tbs, vbs, abs_, dsi, delta, risk_tier, masking_alert),
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

    def get_high_risk_checkins(self, limit: int = 100) -> List[Dict[str, object]]:
        """
        Retrieve check-ins with RED risk tier.

        Parameters
        ----------
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list of dict
            List of high-risk check-in records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM checkins
            WHERE risk_tier = 'RED'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_masking_alerts(self, limit: int = 100) -> List[Dict[str, object]]:
        """
        Retrieve check-ins with masking alerts.

        Parameters
        ----------
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list of dict
            List of check-in records with masking alerts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM checkins
            WHERE masking_alert IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

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
            risk tier distribution.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM checkins")
        total_checkins = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(dsi) FROM checkins")
        avg_dsi = cursor.fetchone()[0] or 0.0

        cursor.execute(
            """
            SELECT risk_tier, COUNT(*) as count
            FROM checkins
            GROUP BY risk_tier
            """
        )
        risk_distribution = {
            row["risk_tier"]: row["count"] for row in cursor.fetchall()
        }

        cursor.execute("SELECT COUNT(*) FROM checkins WHERE masking_alert IS NOT NULL")
        masking_alerts = cursor.fetchone()[0]

        conn.close()

        return {
            "total_checkins": total_checkins,
            "avg_dsi": round(float(avg_dsi), 3),
            "risk_distribution": risk_distribution,
            "masking_alerts": masking_alerts,
        }