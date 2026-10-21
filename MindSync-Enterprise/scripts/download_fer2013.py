"""Download the APPROVED FER2013 dataset into the git-ignored data dir.

Approved source (do not substitute):
    https://www.kaggle.com/datasets/deadskull7/fer2013/data

Downloads via KaggleHub (official Kaggle library). No credentials are
printed or stored here. If Kaggle asks for sign-in / licence acceptance,
do it in the browser once, then re-run this script.

Usage:
    python scripts/download_fer2013.py
    python scripts/download_fer2013.py --verify   # also run the validator

This script NEVER trains a model and never writes model weights.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

APPROVED_HANDLE = "deadskull7/fer2013"
APPROVED_URL = "https://www.kaggle.com/datasets/deadskull7/fer2013/data"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "fer2013")
CSV_NAME = "fer2013.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def download() -> str:
    """Download the approved dataset into data/fer2013/. Returns CSV path."""
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is required. Install it with:\n"
            "    python -m pip install kagglehub\n"
            "(KaggleHub is the official Kaggle download library.)")

    target_csv = os.path.join(DATA_DIR, CSV_NAME)
    if os.path.exists(target_csv):
        log(f"Already present, skipping download: {target_csv}")
        return target_csv

    # KaggleHub refuses a non-empty output dir, so stage in a temp dir
    # and move the extracted files into place afterwards.
    staging = tempfile.mkdtemp(prefix="fer2013_stage_")
    try:
        log(f"Downloading {APPROVED_HANDLE} (approved source: {APPROVED_URL})")
        kagglehub.dataset_download(APPROVED_HANDLE, output_dir=staging)
        os.makedirs(DATA_DIR, exist_ok=True)
        for name in sorted(os.listdir(staging)):
            src = os.path.join(staging, name)
            dst = os.path.join(DATA_DIR, name)
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if not os.path.exists(target_csv):
        raise SystemExit(
            f"Download finished but {target_csv} is missing. "
            "Check the Kaggle dataset page for the expected file name.")
    return target_csv


def verify(csv_path: str) -> int:
    """Run the read-only validator over the downloaded CSV."""
    script = os.path.join(ROOT, "scripts", "verify_fer2013.py")
    report = os.path.join(ROOT, "docs", "fer2013_dataset_report.json")
    cmd = [sys.executable, script, "--csv", csv_path, "--out", report]
    log("Running validator: " + " ".join(cmd))
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="run scripts/verify_fer2013.py after download")
    args = ap.parse_args(argv)

    csv_path = download()
    log(f"CSV ready: {csv_path} ({os.path.getsize(csv_path)} bytes)")
    if args.verify:
        return verify(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
