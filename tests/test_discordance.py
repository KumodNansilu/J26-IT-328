"""
Unit tests for the Discordance Delta calculation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core_fusion_engine.discordance import calculate_discordance_delta, classify_masking


class TestDiscordanceDelta:
    """Test cases for Discordance Delta calculation."""

    def test_honest_consistent(self):
        """Scores close together should have small delta."""
        delta = calculate_discordance_delta(0.30, 0.35, 0.28)
        assert delta == pytest.approx(0.05)  # max(0.35, 0.28) - 0.30 = 0.05

    def test_traditional_masking(self):
        """VBS/ABS high, TBS low -> positive delta > +0.50."""
        delta = calculate_discordance_delta(0.15, 0.80, 0.75)
        assert delta == pytest.approx(0.65)  # max(0.80, 0.75) - 0.15 = 0.65
        assert delta > 0.50

    def test_forced_composure(self):
        """TBS high, VBS/ABS low -> negative delta < -0.50."""
        delta = calculate_discordance_delta(0.85, 0.20, 0.15)
        assert delta == pytest.approx(-0.65)  # max(0.20, 0.15) - 0.85 = -0.65
        assert delta < -0.50

    def test_uses_max_of_vbs_abs(self):
        """Delta should use max(VBS, ABS), not the average."""
        delta = calculate_discordance_delta(0.30, 0.80, 0.20)
        assert delta == pytest.approx(0.50)  # max(0.80, 0.20) - 0.30 = 0.50

    def test_all_equal(self):
        """All scores equal should have delta of 0."""
        delta = calculate_discordance_delta(0.50, 0.50, 0.50)
        assert delta == 0.0

    def test_extreme_values(self):
        """Extreme score differences should be detected."""
        delta = calculate_discordance_delta(0.0, 1.0, 0.0)
        assert delta == 1.0

        delta = calculate_discordance_delta(1.0, 0.0, 0.0)
        assert delta == -1.0


class TestMaskingClassification:
    """Test cases for masking classification."""

    def test_no_masking(self):
        """Small delta should not trigger masking."""
        result = classify_masking(0.10)
        assert result["is_masking"] is False
        assert result["masking_type"] is None

    def test_positive_masking(self):
        """Positive delta > threshold should be Traditional Emotional Masking."""
        result = classify_masking(0.65)
        assert result["is_masking"] is True
        assert result["masking_type"] == "positive"
        assert "Traditional Emotional Masking" in result["alert_message"]

    def test_negative_masking(self):
        """Negative delta < -threshold should be Forced Composure."""
        result = classify_masking(-0.65)
        assert result["is_masking"] is True
        assert result["masking_type"] == "negative"
        assert "Forced Composure" in result["alert_message"]

    def test_exact_threshold(self):
        """Delta exactly at threshold should not trigger masking (strict >)."""
        result = classify_masking(0.50)
        assert result["is_masking"] is False