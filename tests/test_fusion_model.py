"""
Unit tests for the Gated Multimodal Fusion Engine (GMU) model.

Verifies:
    - DSI calculations
    - Dynamic weight shifting during deception
    - Database logging
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from core_fusion_engine.config import MODEL_WEIGHTS_PATH, GATE_HIDDEN_DIM
from core_fusion_engine.discordance import calculate_discordance_delta
from core_fusion_engine.fusion_model import GatedMultimodalFusionEngine
from core_fusion_engine.severity import classify_risk_tier
from hr_dashboard_app.database_manager import DatabaseManager


def load_trained_model() -> GatedMultimodalFusionEngine:
    """Load the trained GMU model weights if available."""
    model = GatedMultimodalFusionEngine(hidden_dim=GATE_HIDDEN_DIM)
    if MODEL_WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


class TestGatedMultimodalFusionEngine:
    """Test cases for the GMU fusion model."""

    def setup_method(self):
        self.model = GatedMultimodalFusionEngine(hidden_dim=GATE_HIDDEN_DIM)

    def test_output_shape(self):
        """Model should output shape (batch, 1)."""
        tbs = torch.tensor([[0.3], [0.5], [0.7]])
        vbs = torch.tensor([[0.4], [0.6], [0.8]])
        abs_ = torch.tensor([[0.2], [0.4], [0.6]])
        delta = torch.tensor([[0.1], [0.1], [0.1]])

        dsi, weights, _ = self.model(tbs, vbs, abs_, delta)
        assert dsi.shape == (3, 1)
        assert weights.shape == (3, 3)

    def test_weights_sum_to_one(self):
        """Gate weights should sum to 1.0 (softmax)."""
        tbs = torch.tensor([[0.3], [0.5]])
        vbs = torch.tensor([[0.4], [0.6]])
        abs_ = torch.tensor([[0.2], [0.4]])
        delta = torch.tensor([[0.1], [0.1]])

        _, weights, _ = self.model(tbs, vbs, abs_, delta)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_dsi_in_range(self):
        """DSI should be in [0, 1]."""
        tbs = torch.tensor([[0.3], [0.5], [0.7]])
        vbs = torch.tensor([[0.4], [0.6], [0.8]])
        abs_ = torch.tensor([[0.2], [0.4], [0.6]])
        delta = torch.tensor([[0.1], [0.1], [0.1]])

        dsi, _, _ = self.model(tbs, vbs, abs_, delta)
        assert torch.all(dsi >= 0.0)
        assert torch.all(dsi <= 1.0)

    def test_predict_returns_tuple(self):
        """predict() should return (dsi, weights, delta)."""
        dsi, weights, delta = self.model.predict(0.3, 0.4, 0.2)
        assert isinstance(dsi, float)
        assert isinstance(weights, list)
        assert len(weights) == 3
        assert isinstance(delta, float)
        assert 0.0 <= dsi <= 1.0
        assert abs(sum(weights) - 1.0) < 1e-5

    def test_dynamic_weight_shifting_during_masking(self):
        """
        During Traditional Emotional Masking (VBS/ABS high, TBS low),
        the trained model should shift weights away from the deceptive
        text modality and toward the truthful video/audio modalities.
        """
        # Use the trained model with hidden_dim matching training
        self.model = load_trained_model()

        # Traditional Emotional Masking: TBS low, VBS/ABS high
        dsi, weights, delta = self.model.predict(0.15, 0.80, 0.75)
        assert delta > 0.50  # Confirms masking scenario

        # The trained model should assign lower weight to text (deceptive)
        # and higher weight to video/audio (truthful modalities)
        assert weights[0] < weights[1] or weights[0] < weights[2]

    def test_dynamic_weight_shifting_during_forced_composure(self):
        """
        During Forced Composure (TBS high, VBS/ABS low),
        the trained model should shift weights toward the text modality.
        """
        # Use the trained model with hidden_dim matching training
        self.model = load_trained_model()

        # Forced Composure: TBS high, VBS/ABS low
        dsi, weights, delta = self.model.predict(0.85, 0.20, 0.15)
        assert delta < -0.50  # Confirms forced composure scenario

        # The trained model should assign higher weight to text (truthful modality)
        assert weights[0] > weights[1] and weights[0] > weights[2]

    def test_gradient_flow(self):
        """Model should support backpropagation."""
        tbs = torch.tensor([[0.3], [0.5]])
        vbs = torch.tensor([[0.4], [0.6]])
        abs_ = torch.tensor([[0.2], [0.4]])
        delta = torch.tensor([[0.1], [0.1]])
        target = torch.tensor([[0.3], [0.5]])

        dsi, _, _ = self.model(tbs, vbs, abs_, delta)
        loss = torch.nn.functional.mse_loss(dsi, target)
        loss.backward()
        assert loss.item() >= 0.0

    def test_state_dict_save_load(self):
        """Model state dict should be saveable and loadable."""
        state_dict = self.model.state_dict()
        new_model = GatedMultimodalFusionEngine(hidden_dim=GATE_HIDDEN_DIM)
        new_model.load_state_dict(state_dict)

        tbs = torch.tensor([[0.3]])
        vbs = torch.tensor([[0.4]])
        abs_ = torch.tensor([[0.2]])
        delta = torch.tensor([[0.1]])

        out1, _, _ = self.model(tbs, vbs, abs_, delta)
        out2, _, _ = new_model(tbs, vbs, abs_, delta)
        assert torch.allclose(out1, out2)


class TestRiskTierClassification:
    """Test cases for risk tier classification."""

    def test_green_tier(self):
        """DSI < 0.45 should be GREEN tier."""
        tier = classify_risk_tier(0.30)
        assert tier.name == "GREEN"

    def test_yellow_tier(self):
        """0.45 <= DSI < 0.70 should be YELLOW tier."""
        tier = classify_risk_tier(0.55)
        assert tier.name == "YELLOW"

    def test_red_tier(self):
        """DSI >= 0.70 should be RED tier."""
        tier = classify_risk_tier(0.80)
        assert tier.name == "RED"

    def test_boundary_yellow(self):
        """DSI exactly 0.45 should be YELLOW tier."""
        tier = classify_risk_tier(0.45)
        assert tier.name == "YELLOW"

    def test_boundary_red(self):
        """DSI exactly 0.70 should be RED tier."""
        tier = classify_risk_tier(0.70)
        assert tier.name == "RED"


class TestDatabaseLogging:
    """Test cases for database logging."""

    def setup_method(self):
        # Use a temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_audit_logs.db"
        self.db = DatabaseManager(self.db_path)

    def teardown_method(self):
        # Clean up
        if self.db_path.exists():
            self.db_path.unlink()

    def test_save_and_retrieve_checkin(self):
        """Check-in should be saved and retrievable."""
        record_id = self.db.save_checkin(
            employee_id="EMP001",
            tbs=0.30,
            vbs=0.35,
            abs_=0.28,
            dsi=0.31,
            delta=0.05,
            risk_tier="GREEN",
        )
        assert record_id > 0

        checkins = self.db.get_checkins(employee_id="EMP001")
        assert len(checkins) == 1
        assert checkins[0]["employee_id"] == "EMP001"
        assert checkins[0]["dsi"] == pytest.approx(0.31)
        assert checkins[0]["risk_tier"] == "GREEN"

    def test_save_checkin_with_masking_alert(self):
        """Check-in with masking alert should be saved."""
        record_id = self.db.save_checkin(
            employee_id="EMP002",
            tbs=0.15,
            vbs=0.80,
            abs_=0.75,
            dsi=0.77,
            delta=0.65,
            risk_tier="RED",
            masking_alert="Traditional Emotional Masking (Fake Happy Text)",
        )
        assert record_id > 0

        masking_alerts = self.db.get_masking_alerts()
        assert len(masking_alerts) == 1
        assert masking_alerts[0]["employee_id"] == "EMP002"

    def test_validate_login(self):
        """Pre-seeded HR login should validate."""
        user = self.db.validate_login("admin@company.com", "admin123")
        assert user is not None
        assert user["username"] == "admin@company.com"

    def test_validate_login_invalid(self):
        """Invalid credentials should return None."""
        user = self.db.validate_login("admin@company.com", "wrongpassword")
        assert user is None

    def test_get_high_risk_checkins(self):
        """High-risk check-ins should be filterable."""
        self.db.save_checkin(
            employee_id="EMP001", tbs=0.3, vbs=0.35, abs_=0.28,
            dsi=0.31, delta=0.05, risk_tier="GREEN",
        )
        self.db.save_checkin(
            employee_id="EMP002", tbs=0.15, vbs=0.80, abs_=0.75,
            dsi=0.77, delta=0.65, risk_tier="RED",
        )

        high_risk = self.db.get_high_risk_checkins()
        assert len(high_risk) == 1
        assert high_risk[0]["employee_id"] == "EMP002"

    def test_summary_stats(self):
        """Summary stats should be computed correctly."""
        self.db.save_checkin(
            employee_id="EMP001", tbs=0.3, vbs=0.35, abs_=0.28,
            dsi=0.31, delta=0.05, risk_tier="GREEN",
        )
        self.db.save_checkin(
            employee_id="EMP002", tbs=0.15, vbs=0.80, abs_=0.75,
            dsi=0.77, delta=0.65, risk_tier="RED",
        )

        stats = self.db.get_summary_stats()
        assert stats["total_checkins"] == 2
        assert stats["avg_dsi"] == pytest.approx(0.54, abs=0.01)
        assert stats["risk_distribution"] == {"GREEN": 1, "RED": 1}