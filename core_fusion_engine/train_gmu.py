"""
Training script for the GMU (Gated Multimodal Unit) fusion model.

This script trains the GMU on the synthetic check-in dataset
(data/checkin_dataset.csv) to learn the mapping from the three
unimodal scores (TBS, VBS, ABS) to the ground-truth DSI.
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .fusion_model import GMUFusionModel

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the check-in dataset and extract the unimodal scores and DSI.

    Parameters
    ----------
    csv_path : str
        Path to the CSV dataset.

    Returns
    -------
    tuple of np.ndarray
        (tbs, vbs, abs_, dsi) arrays.
    """
    df = pd.read_csv(csv_path)

    tbs = df["tbs"].to_numpy(dtype=np.float32).reshape(-1, 1)
    vbs = df["vbs"].to_numpy(dtype=np.float32).reshape(-1, 1)
    abs_ = df["abs"].to_numpy(dtype=np.float32).reshape(-1, 1)
    dsi = df["dsi"].to_numpy(dtype=np.float32).reshape(-1, 1)

    return tbs, vbs, abs_, dsi


def create_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders from the CSV dataset.

    Parameters
    ----------
    csv_path : str
        Path to the CSV dataset.
    batch_size : int
        Batch size for the dataloaders.
    val_split : float
        Fraction of the data to hold out for validation.

    Returns
    -------
    tuple of DataLoader
        (train_loader, val_loader).
    """
    tbs, vbs, abs_, dsi = load_dataset(csv_path)

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
        torch.tensor(dsi[train_idx]),
    )
    val_tensors = (
        torch.tensor(tbs[val_idx]),
        torch.tensor(vbs[val_idx]),
        torch.tensor(abs_[val_idx]),
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
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 16,
) -> GMUFusionModel:
    """
    Train the GMU fusion model.

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
        Hidden dimension of the GMU.

    Returns
    -------
    GMUFusionModel
        The trained model.
    """
    model = GMUFusionModel(hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for tbs_b, vbs_b, abs_b, dsi_b in train_loader:
            optimizer.zero_grad()
            pred = model(tbs_b, vbs_b, abs_b)
            loss = criterion(pred, dsi_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * tbs_b.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tbs_b, vbs_b, abs_b, dsi_b in val_loader:
                pred = model(tbs_b, vbs_b, abs_b)
                loss = criterion(pred, dsi_b)
                val_loss += loss.item() * tbs_b.size(0)

        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train the GMU model on the synthetic dataset and save it.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "checkin_dataset.csv")
    model_save_path = os.path.join(project_root, "models", "gmu_fusion.pt")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    train_loader, val_loader = create_dataloaders(csv_path)
    model = train_model(train_loader, val_loader)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()