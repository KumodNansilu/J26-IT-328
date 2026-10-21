"""Train FerCNN on FER2013 CSV. Full runs need explicit approval.

Usage (after dataset source approved):
  python -m ai_services.video_facial_emotion_behavior.emotion_recognition.train_fer \
      --csv data/fer2013.csv --out models/fer2013_cnn_best.pt --epochs 50
Smoke test (allowed now): --smoke (tiny subset, max 2 epochs, CPU).
Saves best checkpoint by val loss + meta JSON (label order, seed, config).
PrivateTest never touched here — see evaluation/evaluate_model.py.
"""
from __future__ import annotations

import argparse
import json
import os


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_training(*, csv_path: str, out_path: str, epochs: int = 50,
                 batch_size: int = 64, lr: float = 1e-3, patience: int = 7,
                 seed: int = 42, smoke: bool = False) -> dict:
    """Train and save best checkpoint. Returns summary dict."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from .cnn_model import build_model
    from .fer2013 import (USAGE_TRAIN, load_fer_csv, split_train_val)
    from .labels import EMOTIONS, NUM_EMOTIONS

    if smoke:
        epochs = min(epochs, 2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    records, report = load_fer_csv(csv_path)
    train, val = split_train_val(records, val_fraction=0.1, seed=seed)
    if smoke:  # tiny subset: <= 70 train, <= 14 val, proves plumbing only
        train, val = train[:70], val[:14]
    if not train or not val:
        raise ValueError(f"Not enough data: train={len(train)} val={len(val)}.")

    class _Ds(Dataset):
        def __init__(self, recs, augment=False):
            self.recs = recs
            self.augment = augment

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            import cv2

            r = self.recs[i]
            img = r.image
            if self.augment:
                h, w = img.shape
                dx = int(np.random.randint(-2, 3))
                dy = int(np.random.randint(-2, 3))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)
                if np.random.rand() < 0.5:
                    img = cv2.flip(img, 1)
            x = img.astype(np.float32) / 255.0
            return torch.from_numpy(x).unsqueeze(0), r.label

    counts = [sum(1 for r in train if r.label == c) for c in range(NUM_EMOTIONS)]
    total = max(sum(counts), 1)
    weights = torch.tensor([total / max(c, 1) for c in counts],
                           dtype=torch.float32)
    weights = weights / weights.sum() * NUM_EMOTIONS
    train_dl = DataLoader(_Ds(train, True), batch_size=batch_size,
                          shuffle=True)
    val_dl = DataLoader(_Ds(val, False), batch_size=batch_size)
    device = "cpu" if smoke else _device()
    model = build_model(NUM_EMOTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    best_val, best_state, bad = float("inf"), None, 0
    history = []
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vloss += loss_fn(model(xb), yb).item() * len(xb)
                n += len(xb)
        vloss /= max(n, 1)
        history.append(vloss)
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"state_dict": best_state}, out_path)

    # Fixed, shared metadata path produced by this module (see labels.py).
    # Training and inference MUST stay in sync here and in predict.py.
    meta_path = os.path.join(
        os.path.dirname(out_path) or ".",
        "fer2013_cnn_meta.json",
    )
    meta = {
        "label_order": list(EMOTIONS),
        "input": [1, 48, 48],
        "normalize": "x/255.0",
        "seed": seed,
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "smoke": smoke,
        "class_counts": {EMOTIONS[c]: counts[c] for c in range(NUM_EMOTIONS)},
        "torch": torch.__version__,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return {
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "train_n": len(train),
        "val_n": len(val),
        "out": out_path,
        "meta": meta_path,
        "report": report.message(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="models/fer2013_cnn_best.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    summary = run_training(csv_path=args.csv, out_path=args.out,
                           epochs=args.epochs, batch_size=args.batch,
                           smoke=args.smoke)
    print(json.dumps({k: v for k, v in summary.items() if k != "report"},
                     indent=2))
    print(summary["report"])


if __name__ == "__main__":
    main()
