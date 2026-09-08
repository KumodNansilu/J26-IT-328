"""
Training script for the Gated Multimodal Fusion Engine (GMU).

Generates 1,000 synthetic check-in samples covering honest, traditional
masking, and forced composure scenarios, then trains the PyTorch model
for 40 epochs using Adam optimizer and MSE loss.

Saves the trained weights to `core_fusion_engine/gmu_fusion_model.pth`.
"""

import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_SAMPLES,
    TRAIN_BATCH_SIZE,
    GATE_HIDDEN_DIM,
    MODEL_WEIGHTS_PATH,
)
from .discordance import calculate_discordance_delta
from .fusion_model import GatedMultimodalFusionEngine

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int = TRAIN_SAMPLES,
) -> pd.DataFrame:
    """
    Generate synthetic check-in samples covering 4 edge-case scenarios.

    Scenarios:
        1. Honest / Consistent: TBS ≈ VBS ≈ ABS (no masking)
        2. Traditional Emotional Masking: VBS/ABS high, TBS low (Δ > +0.50)
        3. Forced Composure: TBS high, VBS/ABS low (Δ < -0.50)
        4. Mixed / Moderate: moderate scores with small discordance

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: tbs, vbs, abs, dsi, delta, scenario.
    """
    rows = []

    # Split samples across scenarios
    n_honest = int(n_samples * 0.40)
    n_masking = int(n_samples * 0.25)
    n_forced = int(n_samples * 0.25)
    n_mixed = n_samples - n_honest - n_masking - n_forced

    # Scenario 1: Honest / Consistent
    for _ in range(n_honest):
        base = np.random.uniform(0.1, 0.9)
        noise = np.random.uniform(-0.05, 0.05, 3)
        tbs = np.clip(base + noise[0], 0.0, 1.0)
        vbs = np.clip(base + noise[1], 0.0, 1.0)
        abs_ = np.clip(base + noise[2], 0.0, 1.0)
        delta = calculate_discordance_delta(tbs, vbs, abs_)
        dsi = (tbs + vbs + abs_) / 3.0
        rows.append((tbs, vbs, abs_, dsi, delta, "honest"))

    # Scenario 2: Traditional Emotional Masking (Fake Happy Text)
    # Δ = max(VBS, ABS) - TBS > +0.50
    # Non-verbal cues (VBS/ABS) are HIGH, text (TBS) is LOW
    for _ in range(n_masking):
        vbs = np.random.uniform(0.60, 0.95)      # Face shows positive
        abs_ = np.random.uniform(0.60, 0.95)     # Voice shows positive
        tbs = np.random.uniform(0.05, 0.40)      # Text says "I'm fine" (masked)
        delta = calculate_discordance_delta(tbs, vbs, abs_)
        # DSI reflects the truthful non-verbal cues (deceptive text down-weighted)
        dsi = (vbs + abs_) / 2.0
        rows.append((tbs, vbs, abs_, dsi, delta, "masking"))

    # Scenario 3: Forced Composure (Distressed Text, Calm Exterior)
    # Δ = max(VBS, ABS) - TBS < -0.50
    # Text (TBS) is HIGH, non-verbal cues (VBS/ABS) are LOW
    for _ in range(n_forced):
        tbs = np.random.uniform(0.60, 0.95)      # Text reveals distress
        vbs = np.random.uniform(0.05, 0.40)      # Face is calm (forced)
        abs_ = np.random.uniform(0.05, 0.40)     # Voice is calm (forced)
        delta = calculate_discordance_delta(tbs, vbs, abs_)
        # DSI reflects the truthful text modality
        dsi = tbs
        rows.append((tbs, vbs, abs_, dsi, delta, "forced_composure"))

    # Scenario 4: Mixed / Moderate
    for _ in range(n_mixed):
        tbs = np.random.uniform(0.30, 0.70)
        vbs = np.random.uniform(0.30, 0.70)
        abs_ = np.random.uniform(0.30, 0.70)
        delta = calculate_discordance_delta(tbs, vbs, abs_)
        dsi = (tbs + vbs + abs_) / 3.0
        rows.append((tbs, vbs, abs_, dsi, delta, "mixed"))

    df = pd.DataFrame(
        rows,
        columns=["tbs", "vbs", "abs", "dsi", "delta", "scenario"],
    )
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def create_dataloaders(
    df: pd.DataFrame,
    batch_size: int = TRAIN_BATCH_SIZE,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders from the synthetic data.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic dataset with columns tbs, vbs, abs, dsi, delta.
    batch_size : int
        Batch size for the dataloaders.
    val_split : float
        Fraction of the data to hold out for validation.

    Returns
    -------
    tuple of DataLoader
        (train_loader, val_loader).
    """
    tbs = df["tbs"].to_numpy(dtype=np.float32).reshape(-1, 1)
    vbs = df["vbs"].to_numpy(dtype=np.float32).reshape(-1, 1)
    abs_ = df["abs"].to_numpy(dtype=np.float32).reshape(-1, 1)
    delta = df["delta"].to_numpy(dtype=np.float32).reshape(-1, 1)
    dsi = df["dsi"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Shuffle and split
    n = len(tbs)
    indices = np.random.permutation(n)
    n_val = int(n * val_split)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # Build tensors
    train_tensors = (
        torch.tensor(tbs[train_idx]),
        torch.tensor(vbs[train_idx]),
        torch.tensor(abs_[train_idx]),
        torch.tensor(delta[train_idx]),
        torch.tensor(dsi[train_idx]),
    )
    val_tensors = (
        torch.tensor(tbs[val_idx]),
        torch.tensor(vbs[val_idx]),
        torch.tensor(abs_[val_idx]),
        torch.tensor(delta[val_idx]),
        torch.tensor(dsi[val_idx]),
    )

    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    hidden_dim: int = 16,
) -> GatedMultimodalFusionEngine:
    """
    Train the Gated Multimodal Fusion Engine.

    Uses the Adam optimizer with MSE loss and a ReduceLROnPlateau
    scheduler. The best (lowest validation loss) checkpoint is retained.

    Parameters
    ----------
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    hidden_dim : int
        Hidden dimension of the gating MLP.

    Returns
    -------
    GatedMultimodalFusionEngine
        The trained model (best validation checkpoint).
    """
    model = GatedMultimodalFusionEngine(hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for tbs_b, vbs_b, abs_b, delta_b, dsi_b in train_loader:
            optimizer.zero_grad()
            pred, _, _ = model(tbs_b, vbs_b, abs_b, delta_b)
            loss = criterion(pred, dsi_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * tbs_b.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tbs_b, vbs_b, abs_b, delta_b, dsi_b in val_loader:
                pred, _, _ = model(tbs_b, vbs_b, abs_b, delta_b)
                loss = criterion(pred, dsi_b)
                val_loss += loss.item() * tbs_b.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # Keep the best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Generate synthetic data, train the GMU model, and save weights.
    """
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(TRAIN_SAMPLES)
    print(f"Generated {len(df)} samples across scenarios:")
    print(df["scenario"].value_counts().to_string())

    # Save the generated dataset to CSV for reference
    from .config import DATA_DIR
    dataset_path = DATA_DIR / "checkin_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(df)

    print(f"Training for {TRAIN_EPOCHS} epochs...\n")
    model = train_model(
        train_loader,
        val_loader,
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        hidden_dim=GATE_HIDDEN_DIM,
    )

    # Save model weights
    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"\nModel saved to {MODEL_WEIGHTS_PATH}")

    # Quick verification
    model.eval()
    test_cases = [
        (0.30, 0.35, 0.28, "Honest / Consistent"),
        (0.15, 0.80, 0.75, "Traditional Emotional Masking"),
        (0.85, 0.20, 0.15, "Forced Composure"),
    ]
    print("\nVerification predictions:")
    for tbs, vbs, abs_, label in test_cases:
        dsi, weights, delta = model.predict(tbs, vbs, abs_)
        print(
            f"  {label:30s} | TBS={tbs:.2f} VBS={vbs:.2f} ABS={abs_:.2f} "
            f"| Δ={delta:+.2f} | DSI={dsi:.3f} | "
            f"W=[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}]"
        )


if __name__ == "__main__":
    main()