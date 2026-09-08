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

from .discordance import calculate_discordance_delta


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

        return (
            dsi,
            [float(w) for w in weights.squeeze(0)],
            float(delta),
        )