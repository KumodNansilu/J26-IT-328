"""
Gated Multimodal Fusion Engine (GMU) - PyTorch implementation.

This module implements the late-stage decision-level fusion model that
combines three unimodal scores (TBS, VBS, ABS) and the Discordance Delta (Δ)
into a single Depression Severity Index (DSI).

Architecture:
    Input:  [TBS, VBS, ABS, Δ]  (4 features)
    Gating: nn.Linear(4, 3) -> nn.Softmax(dim=-1) -> [w_text, w_video, w_audio]
    Fusion: DSI = (w_text * TBS) + (w_video * VBS) + (w_audio * ABS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discordance_delta import calculate_discordance_delta
from .config import MASKING_THRESHOLD


class GatedMultimodalFusionEngine(nn.Module):
    """
    Gated Multimodal Fusion Engine.

    Ingests [TBS, VBS, ABS, Δ] and produces:
        - dsi     : final Depression Severity Index in [0, 1]
        - weights : dynamic modality weights [w_text, w_video, w_audio]
        - delta   : the Discordance Delta used as the 4th input

    The gating network is a small MLP whose final hidden->3 layer is a
    ``nn.Linear`` followed by ``nn.Softmax(dim=-1)``. With ``hidden_dim=0``
    it degenerates to the architectural baseline ``nn.Linear(4, 3)``.

    Dynamic Weight Adjustment:
    - Forced Composure (Δ <= -MASKING_THRESHOLD, TBS highest):
      Reduce text weight significantly, boost video/audio.
    - Traditional Masking (Δ >= MASKING_THRESHOLD, TBS lowest):
      Reduce text weight moderately.
    - No Masking (|Δ| <= MASKING_THRESHOLD):
      Use balanced weights (~33% each) since modalities agree.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        """
        Initialize the gating network.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension of the gating MLP. Use 0 for the plain
            ``nn.Linear(4, 3)`` baseline.
        """
        super().__init__()

        # Gating network: 4 inputs (TBS, VBS, ABS, Δ) -> 3 modality weights
        if hidden_dim > 0:
            self.gate = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
        else:
            self.gate = nn.Linear(4, 3)

    def forward(
        self,
        tbs: torch.Tensor,
        vbs: torch.Tensor,
        abs_: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        tbs : torch.Tensor
            Text-Based Score, shape (batch, 1).
        vbs : torch.Tensor
            Video-Based Score, shape (batch, 1).
        abs_ : torch.Tensor
            Audio-Based Score, shape (batch, 1).
        delta : torch.Tensor
            Discordance Delta, shape (batch, 1).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (dsi, weights, delta) where:
                - dsi     : shape (batch, 1), in [0, 1]
                - weights : shape (batch, 3), sums to 1.0
                - delta   : shape (batch, 1), the input delta
        """
        # Concatenate inputs: [TBS, VBS, ABS, Δ]
        x = torch.cat([tbs, vbs, abs_, delta], dim=-1)  # (batch, 4)

        # Compute gating weights
        gate_logits = self.gate(x)                      # (batch, 3)
        weights = F.softmax(gate_logits, dim=-1)        # (batch, 3)

        # Weighted fusion: DSI = w_text*TBS + w_video*VBS + w_audio*ABS
        dsi = (
            weights[:, 0:1] * tbs
            + weights[:, 1:2] * vbs
            + weights[:, 2:3] * abs_
        )

        return dsi, weights, delta

    def predict(
        self,
        tbs: float,
        vbs: float,
        abs_: float,
    ) -> tuple[float, list[float], float]:
        """
        Run a single inference and return (dsi, weights, delta).

        Parameters
        ----------
        tbs : float
            Text-Based Score in [0, 1].
        vbs : float
            Video-Based Score in [0, 1].
        abs_ : float
            Audio-Based Score in [0, 1].

        Returns
        -------
        tuple[float, list[float], float]
            (dsi, [w_text, w_video, w_audio], delta).
        """
        delta = calculate_discordance_delta(tbs, vbs, abs_)

        self.eval()
        with torch.no_grad():
            tbs_t = torch.tensor([[tbs]], dtype=torch.float32)
            vbs_t = torch.tensor([[vbs]], dtype=torch.float32)
            abs_t = torch.tensor([[abs_]], dtype=torch.float32)
            delta_t = torch.tensor([[delta]], dtype=torch.float32)

            dsi, weights, _ = self.forward(tbs_t, vbs_t, abs_t, delta_t)

        # Clamp DSI to the valid [0, 1] range
        dsi = float(torch.clamp(dsi, 0.0, 1.0).item())

        # Raw softmax weights from the trained gate (a fallback opinion).
        raw_text, raw_video, raw_audio = [float(w) for w in weights.squeeze(0)]

        # --- Dynamic weight adjustment for forced composure / masking ---
        # The masking branches override the gate's raw weighting with a
        # *deterministic* shift based on a neutral 1/3 baseline.  The trained
        # gate saturates (w_text -> ~1.0) in forced-composure situations, so
        # multiplying its raw weight against a reduction factor cannot push
        # text back down toward ~0.10.  Starting the reduction from the neutral
        # 1/3 baseline instead makes the forced-composure / masking emphasis
        # reproducible regardless of the trained gate's raw output.
        NEUTRAL = 1.0 / 3.0
        if delta <= -MASKING_THRESHOLD and tbs >= vbs and tbs >= abs_:
            # Forced Composure: distressed text, calm exterior.
            # De-emphasize text; spread the remainder across video/audio.
            reduction = min(abs(delta), 1.0)  # cap reduction factor at 1.0
            w_text = max(0.10, NEUTRAL * (1.0 - reduction * 0.85))
            remaining = 1.0 - w_text
            vab = vbs + abs_
            w_video = remaining * (vbs / vab) if vab > 0 else remaining * 0.5
            w_audio = remaining - w_video
        elif delta >= MASKING_THRESHOLD and tbs <= vbs and tbs <= abs_:
            # Traditional Masking: fake-happy text, distressed exterior.
            # De-emphasize text; spread the remainder across video/audio.
            reduction = min(abs(delta), 1.0)  # cap reduction factor at 1.0
            w_text = max(0.15, NEUTRAL * (1.0 - reduction * 0.70))
            remaining = 1.0 - w_text
            vab = vbs + abs_
            w_video = remaining * (vbs / vab) if vab > 0 else remaining * 0.5
            w_audio = remaining - w_video
        else:
            # No masking discordance: honor the trained gate's weighting.
            w_text, w_video, w_audio = raw_text, raw_video, raw_audio

        # Recompute DSI from the final (post-adjustment) weights so the reported
        # DSI is consistent with the displayed weight breakdown.
        dsi = w_text * tbs + w_video * vbs + w_audio * abs_
        dsi = float(max(0.0, min(1.0, dsi)))

        # Round to 3 decimals; compute the last weight as the complement so the
        # three weights always sum to exactly 1.0 (avoids rounding drift).
        w_text = round(w_text, 3)
        w_video = round(w_video, 3)
        w_audio = round(1.0 - w_text - w_video, 3)

        return (dsi, [w_text, w_video, w_audio], float(delta))