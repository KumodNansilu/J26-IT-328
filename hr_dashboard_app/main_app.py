"""
Main application entry point for the HR dashboard.

Provides a Streamlit-based interface for monitoring employee
depression severity using the multimodal fusion engine.
"""

import os
import sys
from typing import Dict, List, Optional

import streamlit as st

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core_fusion_engine.config import FusionConfig
from core_fusion_engine.discordance import DiscordanceDetector
from core_fusion_engine.fusion_model import MultimodalFusionEngine
from core_fusion_engine.severity import SeverityClassifier
from hr_dashboard_app.database_manager import DatabaseManager
from hr_dashboard_app.controllers.alert_controller import AlertController
from hr_dashboard_app.ui_components.risk_gauge import render_risk_gauge
from hr_dashboard_app.ui_components.session_table import render_session_table


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Employee Mental Health Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------

@st.cache_resource
def get_fusion_engine() -> MultimodalFusionEngine:
    """Load the trained multimodal fusion engine."""
    config = FusionConfig()
    engine = MultimodalFusionEngine(config)
    engine.load_weights()
    return engine


@st.cache_resource
def get_discordance_detector() -> DiscordanceDetector:
    """Create the discordance detector."""
    return DiscordanceDetector()


@st.cache_resource
def get_severity_classifier() -> SeverityClassifier:
    """Create the severity classifier."""
    return SeverityClassifier()


@st.cache_resource
def get_database() -> DatabaseManager:
    """Create the database manager."""
    return DatabaseManager()


@st.cache_resource
def get_alert_controller() -> AlertController:
    """Create the alert controller."""
    return AlertController(get_database())


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def compute_dsi(
    tbs: float,
    vbs: float,
    abs_: float,
) -> Dict[str, object]:
    """
    Compute the fused DSI and related metadata.

    Parameters
    ----------
    tbs : float
        Text-based score (0-100).
    vbs : float
        Vision-based score (0-100).
    abs_ : float
        Audio-based score (0-100).

    Returns
    -------
    dict
        Dictionary with dsi, severity, and discordance flag.
    """
    engine = get_fusion_engine()
    detector = get_discordance_detector()
    classifier = get_severity_classifier()

    # Fuse the three modalities
    dsi = engine.predict(tbs, vbs, abs_)

    # Classify severity
    severity = classifier.classify(dsi)

    # Check for discordance
    discordance = detector.detect(tbs, vbs, abs_)

    return {
        "dsi": dsi,
        "severity": severity,
        "discordance": discordance,
    }


def process_checkin(
    employee_id: str,
    tbs: float,
    vbs: float,
    abs_: float,
) -> Dict[str, object]:
    """
    Process a check-in: compute DSI, save record, and raise alerts.

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

    Returns
    -------
    dict
        Dictionary with the computed results and saved record ID.
    """
    db = get_database()
    alert_controller = get_alert_controller()

    result = compute_dsi(tbs, vbs, abs_)

    # Save the check-in record
    record_id = db.save_checkin(
        employee_id=employee_id,
        tbs=tbs,
        vbs=vbs,
        abs_=abs_,
        dsi=result["dsi"],
        severity=result["severity"].name,
    )

    # Raise alert if severity is moderate or higher
    if result["severity"].name in ("Moderate", "Severe", "Critical"):
        alert_controller.raise_alert(
            employee_id=employee_id,
            dsi=result["dsi"],
            severity=result["severity"].name,
            message=(
                f"Employee {employee_id} shows {result['severity'].name} "
                f"depression risk (DSI={result['dsi']:.1f})."
            ),
        )

    result["record_id"] = record_id
    return result


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title("🧠 HR Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["New Check-in", "Session History", "Alerts", "Summary"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Multimodal Depression Screening")


# ----------------------------------------------------------------------
# Page: New Check-in
# ----------------------------------------------------------------------

if page == "New Check-in":
    st.title("New Employee Check-in")
    st.markdown(
        "Enter the three modality scores to compute the fused "
        "Depression Severity Index (DSI)."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        employee_id = st.text_input("Employee ID", value="EMP001")

    with col2:
        tbs = st.slider(
            "Text-based Score (TBS)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=0.5,
            help="Score from the NLP text analysis service.",
        )

    with col3:
        vbs = st.slider(
            "Vision-based Score (VBS)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.5,
            help="Score from the computer vision service.",
        )

    abs_ = st.slider(
        "Audio-based Score (ABS)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.5,
        help="Score from the speech analysis service.",
    )

    if st.button("Compute DSI", type="primary"):
        if not employee_id.strip():
            st.error("Please enter a valid Employee ID.")
        else:
            with st.spinner("Computing fused DSI..."):
                result = process_checkin(
                    employee_id=employee_id.strip(),
                    tbs=tbs,
                    vbs=vbs,
                    abs_=abs_,
                )

            st.success(f"Check-in saved (Record #{result['record_id']}).")

            # Display the risk gauge
            render_risk_gauge(
                dsi=result["dsi"],
                severity=result["severity"].name,
            )

            if result["discordance"]["is_discordant"]:
                st.warning(
                    "⚠️ **Modality discordance detected.** "
                    "The three modality scores disagree significantly. "
                    "Consider reviewing the individual modality outputs."
                )

            st.markdown("### Modality Scores")
            col1, col2, col3 = st.columns(3)
            col1.metric("TBS", f"{tbs:.1f}")
            col2.metric("VBS", f"{vbs:.1f}")
            col3.metric("ABS", f"{abs_:.1f}")


# ----------------------------------------------------------------------
# Page: Session History
# ----------------------------------------------------------------------

elif page == "Session History":
    st.title("Session History")

    db = get_database()

    employee_filter = st.text_input(
        "Filter by Employee ID (leave empty for all)",
        value="",
    )

    checkins = db.get_checkins(
        employee_id=employee_filter.strip() or None,
        limit=200,
    )

    if not checkins:
        st.info("No check-in records found.")
    else:
        render_session_table(checkins)


# ----------------------------------------------------------------------
# Page: Alerts
# ----------------------------------------------------------------------

elif page == "Alerts":
    st.title("Alert Notifications")

    db = get_database()

    col1, col2 = st.columns([3, 1])

    with col1:
        show_acknowledged = st.checkbox(
            "Show acknowledged alerts",
            value=False,
        )

    with col2:
        if st.button("Refresh"):
            st.rerun()

    alerts = db.get_alerts(
        acknowledged=None if show_acknowledged else False,
        limit=100,
    )

    if not alerts:
        st.info("No alerts to display.")
    else:
        for alert in alerts:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"**{alert['employee_id']}**")
                    st.caption(alert["timestamp"])

                with col2:
                    st.metric("DSI", f"{alert['dsi']:.1f}")

                with col3:
                    st.markdown(f"**{alert['severity']}**")

                with col4:
                    if not alert["acknowledged"]:
                        if st.button(
                            "Acknowledge",
                            key=f"ack_{alert['id']}",
                        ):
                            db.acknowledge_alert(alert["id"])
                            st.rerun()
                    else:
                        st.caption("✅ Acknowledged")

                st.markdown(alert["message"])


# ----------------------------------------------------------------------
# Page: Summary
# ----------------------------------------------------------------------

elif page == "Summary":
    st.title("Dashboard Summary")

    db = get_database()
    stats = db.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Check-ins", stats["total_checkins"])
    col2.metric("Average DSI", f"{stats['avg_dsi']:.1f}")
    col3.metric("Pending Alerts", stats["pending_alerts"])

    severity_counts = stats["severity_distribution"]
    col4.metric(
        "High-Risk Cases",
        severity_counts.get("Severe", 0) + severity_counts.get("Critical", 0),
    )

    st.markdown("### Severity Distribution")

    if severity_counts:
        chart_data = {
            "Severity": list(severity_counts.keys()),
            "Count": list(severity_counts.values()),
        }
        st.bar_chart(chart_data, x="Severity", y="Count")
    else:
        st.info("No check-in data available yet.")


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Multimodal Depression Screening System")