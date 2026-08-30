"""
Alert controller for the HR Dashboard.

Evaluates check-in sessions against risk thresholds and discordance
criteria, generating alerts and audit log entries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core_fusion_engine.config import (
    RED_TIER_THRESHOLD,
    YELLOW_TIER_THRESHOLD,
    MASKING_THRESHOLD,
)
from core_fusion_engine.discordance import calculate_discordance_delta, classify_masking
from core_fusion_engine.severity import classify_risk_tier
from hr_dashboard_app.database_manager import DatabaseManager


@dataclass
class Alert:
    """Represents an alert generated for a check-in session."""

    employee_id: str
    session_id: int
    alert_type: str  # "HIGH_RISK", "DISCORDANCE", "ESCALATION"
    severity: str    # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    message: str
    dsi: float
    timestamp: str = ""


@dataclass
class AlertController:
    """
    Evaluates sessions and generates alerts.

    Alert conditions:
        1. HIGH_RISK: DSI >= RED_TIER_THRESHOLD
        2. DISCORDANCE: |Δ| > MASKING_THRESHOLD
        3. ESCALATION: Consecutive high-risk sessions >= escalation_count
    """

    db: DatabaseManager
    escalation_count: int = 3

    def evaluate_session(
        self,
        employee_id: str,
        tbs: float,
        vbs: float,
        abs_: float,
        dsi: float,
    ) -> List[Alert]:
        """
        Evaluate a single check-in session and generate alerts.

        Args:
            employee_id: Employee identifier
            tbs: Text-Based Score (0-1)
            vbs: Video-Based Score (0-1)
            abs_: Audio-Based Score (0-1)
            dsi: Computed Depression Severity Index (0-1)

        Returns:
            List of generated alerts.
        """
        alerts: List[Alert] = []

        # Compute discordance
        delta = calculate_discordance_delta(tbs, vbs, abs_)
        masking = classify_masking(delta)

        # Classify risk tier
        risk_tier = classify_risk_tier(dsi)

        # Insert session into database
        session_id = self.db.save_checkin(
            employee_id=employee_id,
            tbs=tbs,
            vbs=vbs,
            abs_=abs_,
            dsi=dsi,
            delta=delta,
            risk_tier=risk_tier.name,
            masking_alert=masking["alert_message"] if masking["is_masking"] else None,
        )

        # Alert 1: High risk
        if dsi >= RED_TIER_THRESHOLD:
            alert = Alert(
                employee_id=employee_id,
                session_id=session_id,
                alert_type="HIGH_RISK",
                severity="HIGH",
                message=(
                    f"High depression risk detected. DSI={dsi:.3f} "
                    f"({risk_tier.name} TIER)."
                ),
                dsi=dsi,
            )
            alerts.append(alert)

        # Alert 2: Discordance / Masking
        if masking["is_masking"]:
            alert = Alert(
                employee_id=employee_id,
                session_id=session_id,
                alert_type="DISCORDANCE",
                severity="MEDIUM",
                message=masking["alert_message"],
                dsi=dsi,
            )
            alerts.append(alert)

        # Alert 3: Escalation (consecutive high-risk sessions)
        recent_sessions = self.db.get_checkins(
            employee_id=employee_id, limit=self.escalation_count
        )
        if len(recent_sessions) >= self.escalation_count:
            consecutive_high_risk = all(
                s["risk_tier"] == "RED" for s in recent_sessions
            )
            if consecutive_high_risk:
                alert = Alert(
                    employee_id=employee_id,
                    session_id=session_id,
                    alert_type="ESCALATION",
                    severity="CRITICAL",
                    message=(
                        f"Escalation: {self.escalation_count} consecutive "
                        f"high-risk sessions detected. Immediate HR intervention required."
                    ),
                    dsi=dsi,
                )
                alerts.append(alert)

        return alerts

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Fetch recent high-risk and discordant sessions as alerts."""
        high_risk = self.db.get_high_risk_checkins()
        masking = self.db.get_masking_alerts()

        # Merge and deduplicate by session ID
        sessions = {s["id"]: s for s in high_risk + masking}
        return list(sessions.values())