# -*- coding: utf-8 -*-
"""FER2013 CNN training entry point (Step-4).

Trains the custom FerCNN (cnn_model.py) on the Kaggle fer2013.csv layout
and writes:
  * <out_path>                     -> best-by-validation-loss weights
  * <out_dir>/fer2013_cnn_meta.json -> metadata consumed by predict.py

Design rules honoured here:
  * Never downloads anything. The CSV path is supplied by the caller.
  * The locked PrivateTest rows are NEVER read for training or validation.
  * Training and inference share labels.py / cnn_model.py, and the metadata
    written here is the exact contract predict.py validates.
  * Reports measured numbers only - nothing is hard-coded or invented.

Usage:
  python -m ai_services.video_facial_emotion_behavior.emotion_recognition.train_fer \
      --csv data/fer2013/fer2013.csv --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from .cnn_model import MODEL_INPUT_SIZE, build_model
from .fer2013 import load_fer_csv, split_train_val
from .labels import DEFAULT_CKPT, DEFAULT_META, EMOTIONS, NUM_EMOTIONS

# Smoke mode caps: CPU only, <=2 epochs, tiny subset (see scripts/smoke_cnn.py).
SMOKE_MAX_EPOCHS = 2
SMOKE_MAX_SAMPLES = 200

# Repository root: <root>/ai_services/video_facial_emotion_behavior/emotion_recognition/
ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str) -> str:
    """Absolute path; relative paths are anchored at the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def meta_path_for(out_path: str) -> str:
    """Metadata path that predict.py will look for next to a checkpoint.

    predict.load_model() resolves the metadata as
    ``<dirname(ckpt)>/<basename(DEFAULT_META)>`` - this helper mirrors that
    exactly so training and inference can never drift apart.
    """
    return os.path.join(os.path.dirname(os.path.abspath(out_path)),
                        os.path.basename(DEFAULT_META))


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch so a run is reproducible."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FerDataset:
    """Tensor dataset over FerRecords: 48x48 uint8 gray -> float 0..1.

    Normalisation is ``x/255.0`` - identical to cnn_model.preprocess_crop(),
    so training and inference see the same numeric range.
    """

    def __init__(self, records):
        import numpy as np
        import torch

        self.records = list(records or [])
        if not self.records:
            raise ValueError("FerDataset needs at least one record.")
        imgs = np.stack([np.asarray(r.image, dtype=np.uint8)
                         for r in self.records])
        self.x = torch.from_numpy(
            (imgs.astype("float32") / 255.0).reshape(-1, 1, MODEL_INPUT_SIZE,
                                                   MODEL_INPUT_SIZE))
        self.y = torch.tensor([int(r.label) for r in self.records],
                              dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _class_counts(records) -> dict:
    """{emotion_name: count} for the records actually used."""
    counts: dict = {}
    for r in records or []:
        name = EMOTIONS[int(r.label)]
        counts[name] = counts.get(name, 0) + 1
    return counts


def _subset(records, max_n: int, seed: int):
    """Deterministic class-balanced-ish subset used by smoke mode."""
    if len(records) <= max_n:
        return list(records)
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    return pool[:max_n]


def run_training(*, csv_path: str, out_path: str = DEFAULT_CKPT,
                 epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
                 val_fraction: float = 0.1, seed: int = 42,
                 smoke: bool = False, num_workers: int = 0,
                 device: str = "", meta_path: str = "",
                 patience: int = 0, weight_decay: float = 0.0) -> dict:
    """Train the FerCNN and save the best checkpoint + metadata.

    Only ``Usage=Training`` rows are used; a stratified slice of them becomes
    the validation set. PrivateTest rows are never touched here.

    ``smoke=True`` forces CPU, caps epochs at 2 and subsamples the data -
    it is a plumbing check, NOT accuracy evidence.

    ``patience`` > 0 enables early stopping: training stops when the
    validation loss has not improved for ``patience`` epochs. The best
    checkpoint is always the epoch with the lowest validation loss.

    Returns a summary dict (measured values only).
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    csv_path = _resolve(csv_path)
    out_path = _resolve(out_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "FER2013 CSV not found: %s. Download it first "
            "(scripts/download_fer2013.py) - training never downloads."
            % csv_path)

    if smoke:
        device = "cpu"
        epochs = min(int(epochs), SMOKE_MAX_EPOCHS)
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    # --- data ---------------------------------------------------------
    records, report = load_fer_csv(csv_path)
    if not records:
        raise ValueError("No usable rows in %s (kept %d/%d)."
                         % (csv_path, report.kept, report.total_seen))
    train_recs, val_recs = split_train_val(records, val_fraction=val_fraction,
                                            seed=seed)
    if smoke:
        train_recs = _subset(train_recs, SMOKE_MAX_SAMPLES, seed)
        val_recs = _subset(val_recs, max(SMOKE_MAX_SAMPLES // 4, 1), seed + 1)
    if not train_recs or not val_recs:
        raise ValueError(
            "Training needs both train and validation rows "
            "(got %d train / %d val). Check the Usage column."
            % (len(train_recs), len(val_recs)))

    train_ds = FerDataset(train_recs)
    val_ds = FerDataset(val_recs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    # --- model --------------------------------------------------------
    model = build_model(num_classes=NUM_EMOTIONS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=float(weight_decay))

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_run = 0
    epochs_without_improvement = 0
    stopped_early = False
    patience = int(patience)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)
            seen += int(xb.size(0))

        model.eval()
        val_loss, val_n, correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += float(criterion(logits, yb).item()) * xb.size(0)
                correct += int((logits.argmax(dim=1) == yb).sum().item())
                val_n += int(xb.size(0))
        val_loss /= max(val_n, 1)
        val_acc = correct / max(val_n, 1)
        epochs_run = epoch
        history.append({"epoch": epoch,
                        "train_loss": running / max(seen, 1),
                        "val_loss": val_loss,
                        "val_accuracy": val_acc})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        print("epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f"
              % (epoch, epochs, running / max(seen, 1), val_loss, val_acc),
              flush=True)
        # Early stopping is driven ONLY by validation loss; the test set is
        # never consulted here.
        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            print("early stopping: no val_loss improvement for %d epochs "
                  "(best epoch %d)" % (patience, best_epoch), flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint - nothing was run.")

    # --- save ---------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save({"state_dict": best_state}, out_path)

    meta_path = _resolve(meta_path) if meta_path else meta_path_for(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".",
                exist_ok=True)
    meta = {
        "label_order": list(EMOTIONS),
        "input": [1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "normalize": "x/255.0",
        "seed": int(seed),
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "smoke": bool(smoke),
        "class_counts": _class_counts(train_recs),
        "torch": torch.__version__,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    summary = {
        "out": out_path,
        "meta": meta_path,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "epochs_requested": int(epochs),
        "stopped_early": bool(stopped_early),
        "patience": int(patience),
        "weight_decay": float(weight_decay),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "val_fraction": float(val_fraction),
        "train_n": len(train_recs),
        "val_n": len(val_recs),
        "class_counts": _class_counts(train_recs),
        "val_class_counts": _class_counts(val_recs),
        "smoke": bool(smoke),
        "device": device,
        "torch": torch.__version__,
        "seed": int(seed),
        "loader": {
            "total_seen": report.total_seen,
            "kept": report.kept,
            "skipped": report.skipped,
            "problems": report.problems[:20],
        },
        "history": history,
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the FER2013 CNN (no download, no test-set use).")
    ap.add_argument("--csv", required=True,
                    help="path to fer2013.csv (emotion,pixels,Usage)")
    ap.add_argument("--out", default=DEFAULT_CKPT,
                    help="checkpoint output path (default: %s)" % DEFAULT_CKPT)
    ap.add_argument("--meta", default="",
                    help="metadata output path (default: next to --out)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam weight decay (default: %s)" % 0.0)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping patience in epochs (0 = disabled)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="", help="cpu | cuda | '' (auto)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU, <=2 epochs, small subset (plumbing check only)")
    args = ap.parse_args(argv)

    summary = run_training(
        csv_path=args.csv, out_path=args.out, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        val_fraction=args.val_fraction, seed=args.seed, smoke=args.smoke,
        num_workers=args.num_workers, device=args.device,
        meta_path=args.meta)

    printable = {k: v for k, v in summary.items() if k != "history"}
    printable["history"] = summary["history"]
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

