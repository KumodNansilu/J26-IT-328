"""
GMU (Gated Multimodal Unit) Fusion Model.

This module implements a lightweight Gated Multimodal Unit that fuses the
three unimodal sentiment scores (TBS, VBS, ABS) into a single
Decision Sentiment Index (DSI) in [0, 1].

The GMU learns modality-specific gating weights so that the most reliable
modality dominates the fused output for each sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMUFusionModel(nn.Module):
    """
    Gated Multimodal Unit for fusing TBS, VBS and ABS into a DSI.

    Architecture
    ------------
    - Three modality encoders (one per modality), each a small MLP.
    - A gating network that produces per-modality weights.
    - A weighted sum of the encoded modalities followed by a sigmoid
      to produce the final DSI in [0, 1].
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 1,
    ) -> None:
        """
        Initialize the GMU fusion model.

        Parameters
        ----------
        input_dim : int
            Dimensionality of each unimodal score (default 1).
        hidden_dim : int
            Hidden layer size for the encoders and gating network.
        output_dim : int
            Dimensionality of the fused DSI (default 1).
        """
        super().__init__()

        # Modality encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vision_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Final fusion layer
        self.fusion_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        tbs: torch.Tensor,
        vbs: torch.Tensor,
        abs_: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass fusing the three unimodal scores.

        Parameters
        ----------
        tbs : torch.Tensor
            Text-Based Sentiment scores, shape (batch, 1).
        vbs : torch.Tensor
            Visual-Based Sentiment scores, shape (batch, 1).
        abs_ : torch.Tensor
            Audio-Based Sentiment scores, shape (batch, 1).

        Returns
        -------
        torch.Tensor
            Fused DSI scores in [0, 1], shape (batch, 1).
        """
        # Encode each modality
        h_text = self.text_encoder(tbs)
        h_vision = self.vision_encoder(vbs)
        h_audio = self.audio_encoder(abs_)

        # Compute gating weights from the concatenated raw scores
        concat_raw = torch.cat([tbs, vbs, abs_], dim=-1)
        gate_logits = self.gate_network(concat_raw)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, 3)

        # Weighted sum of encoded modalities
        fused = (
            gate_weights[:, 0:1] * h_text
            + gate_weights[:, 1:2] * h_vision
            + gate_weights[:, 2:3] * h_audio
        )

        # Final projection to DSI
        dsi = torch.sigmoid(self.fusion_layer(fused))
        return dsi


def predict_dsi(
    model: GMUFusionModel,
    tbs: float,
    vbs: float,
    abs_: float,
) -> float:
    """
    Run a single inference through the trained GMU model.

    Parameters
    ----------
    model : GMUFusionModel
        A trained GMU fusion model.
    tbs : float
        Text-Based Sentiment score in [0, 1].
    vbs : float
        Visual-Based Sentiment score in [0, 1].
    abs_ : float
        Audio-Based Sentiment score in [0, 1].

    Returns
    -------
    float
        Fused Decision Sentiment Index (DSI) in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        tbs_t = torch.tensor([[tbs]], dtype=torch.float32)
        vbs_t = torch.tensor([[vbs]], dtype=torch.float32)
        abs_t = torch.tensor([[abs_]], dtype=torch.float32)
        dsi = model(tbs_t, vbs_t, abs_t)
    return float(dsi.item())