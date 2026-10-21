"""FER2013 loader (CSV + folder layouts). Source NOT approved yet.

Parses a file/folder YOU supply after approval. Never downloads.
CSV: emotion,pixels,Usage (2304 gray values, 48x48). Folders:
root/{train,test}/<emotion>/*.jpg|png. Label map asserted by tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .labels import EMOTIONS, NUM_EMOTIONS

IMG_SIZE = 48
IMG_PIXELS = IMG_SIZE * IMG_SIZE

# CSV 0..6 = Angry..Neutral, identical to labels.py (tested, not assumed).
FER_CSV_TO_LABELS_INDEX: dict[int, int] = {i: i for i in range(NUM_EMOTIONS)}

USAGE_TRAIN = "Training"
USAGE_PUBLIC = "PublicTest"
USAGE_PRIVATE = "PrivateTest"
KNOWN_USAGES = (USAGE_TRAIN, USAGE_PUBLIC, USAGE_PRIVATE)


@dataclass
class FerRecord:
    """One parsed sample: 48x48 uint8 gray + label index into EMOTIONS."""

    image: object
    label: int
    usage: str
    source: str = ""


@dataclass
class FerLoadReport:
    """Counts + problem log. Bad rows skipped, never crash load."""

    total_seen: int = 0
    kept: int = 0
    skipped: int = 0
    problems: list = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)

    def message(self) -> str:
        return (f"kept {self.kept}/{self.total_seen} "
                f"(skipped {self.skipped}). counts={self.class_counts}")


def _parse_pixels(text: str, *, row_id: str):
    """Parse 'v v v...' into (48,48) uint8. Returns (img|None, problem|None)."""
    import numpy as np

    parts = (text or "").strip().split()
    if len(parts) != IMG_PIXELS:
        return None, f"{row_id}: expected {IMG_PIXELS} pixels, got {len(parts)}"
    try:
        vals = np.array([int(v) for v in parts], dtype=np.int32)
    except ValueError:
        return None, f"{row_id}: non-integer pixel value"
    if vals.min() < 0 or vals.max() > 255:
        return None, f"{row_id}: pixel out of range 0-255"
    return vals.astype("uint8").reshape(IMG_SIZE, IMG_SIZE), None


def load_fer_csv(path: str) -> tuple[list[FerRecord], FerLoadReport]:
    """Load Kaggle fer2013.csv. Skips bad rows with reasons in report."""
    import csv

    report = FerLoadReport()
    records: list[FerRecord] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        if not {"emotion", "pixels", "Usage"} <= cols:
            raise ValueError(
                f"Unexpected CSV header {reader.fieldnames}: need "
                "emotion,pixels,Usage (Kaggle fer2013.csv layout).")
        for i, row in enumerate(reader, start=2):
            report.total_seen += 1
            row_id = f"row {i}"
            try:
                label = int((row.get("emotion") or "").strip())
            except ValueError:
                report.skipped += 1
                report.problems.append(f"{row_id}: bad emotion value")
                continue
            if label not in FER_CSV_TO_LABELS_INDEX:
                report.skipped += 1
                report.problems.append(f"{row_id}: unknown label {label}")
                continue
            usage = (row.get("Usage") or "").strip()
            if usage not in KNOWN_USAGES:
                report.skipped += 1
                report.problems.append(f"{row_id}: unknown Usage '{usage}'")
                continue
            img, prob = _parse_pixels(row.get("pixels") or "", row_id=row_id)
            if prob:
                report.skipped += 1
                report.problems.append(prob)
                continue
            idx = FER_CSV_TO_LABELS_INDEX[label]
            records.append(FerRecord(image=img, label=idx, usage=usage,
                                     source=row_id))
            report.kept += 1
            name = EMOTIONS[idx]
            report.class_counts[name] = report.class_counts.get(name, 0) + 1
    return records, report


def load_fer_folders(root: str) -> tuple[list[FerRecord], FerLoadReport]:
    """Load root/{train,test}/<emotion>/*.(jpg|png|jpeg)."""
    import os

    import cv2

    name_to_idx = {n.lower(): i for i, n in enumerate(EMOTIONS)}
    report = FerLoadReport()
    records: list[FerRecord] = []
    for split in ("train", "test"):
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            report.problems.append(f"missing split dir: {sdir}")
            continue
        for folder in sorted(os.listdir(sdir)):
            fdir = os.path.join(sdir, folder)
            if not os.path.isdir(fdir):
                continue
            idx = name_to_idx.get(folder.lower())
            if idx is None:
                report.problems.append(f"unknown label folder '{folder}'")
                continue
            for fn in sorted(os.listdir(fdir)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fp = os.path.join(fdir, fn)
                report.total_seen += 1
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    report.skipped += 1
                    report.problems.append(f"{fp}: unreadable image")
                    continue
                if img.shape != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_AREA)
                records.append(FerRecord(image=img, label=idx, usage=split,
                                         source=fp))
                report.kept += 1
                report.class_counts[EMOTIONS[idx]] = (
                    report.class_counts.get(EMOTIONS[idx], 0) + 1)
    return records, report


def split_train_val(records, *, val_fraction: float = 0.1, seed: int = 42):
    """Stratified val split from Training rows only. Test never touched."""
    import random

    train_only = [r for r in (records or []) if r.usage == USAGE_TRAIN]
    by_class: dict[int, list] = {}
    for r in train_only:
        by_class.setdefault(r.label, []).append(r)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        group = list(by_class[label])
        rng.shuffle(group)
        n_val = int(round(len(group) * val_fraction))
        n_val = min(max(n_val, 1 if len(group) >= 2 else 0), len(group))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def lock_test_set(records, *, allow_test: bool = False):
    """PrivateTest rows only when explicitly allowed (final eval)."""
    if not allow_test:
        return []
    return [r for r in (records or []) if r.usage == USAGE_PRIVATE]