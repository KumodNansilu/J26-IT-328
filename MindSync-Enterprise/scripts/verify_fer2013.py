"""Validate a downloaded FER2013 CSV against the project's expected layout.

Read-only. Reports structure, partitions, label mapping, and data quality.
Never repairs, relabels, or modifies the dataset.

Usage:
  python scripts/verify_fer2013.py --csv data/fer2013/fer2013.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_services.video_facial_emotion_behavior.emotion_recognition.fer2013 import (  # noqa: E402
    IMG_PIXELS,
    IMG_SIZE,
    KNOWN_USAGES,
    load_fer_csv,
)
from ai_services.video_facial_emotion_behavior.emotion_recognition.labels import (  # noqa: E402
    EMOTIONS,
)


def scan(path):
    """Stream the CSV once and collect structure + quality facts."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        usage_counts, class_counts, usage_class_counts = {}, {}, {}
        labels_seen, usages_seen = set(), set()
        pixel_len_hist, bad_pixels_len = {}, []
        bad_labels, bad_usage = [], []
        empty_fields = non_integer_pixels = 0
        dup_hashes, duplicate_rows, total_rows = {}, 0, 0

        for i, row in enumerate(reader, start=2):
            total_rows += 1
            emo_raw = row.get("emotion")
            pix_raw = row.get("pixels")
            use_raw = row.get("Usage")
            if emo_raw is None or pix_raw is None or use_raw is None:
                empty_fields += 1
                continue

            emo_s = (emo_raw or "").strip()
            use_s = (use_raw or "").strip()
            pix_s = (pix_raw or "").strip()

            if use_s not in KNOWN_USAGES:
                bad_usage.append({"row": i, "value": use_s[:60]})
                continue
            usages_seen.add(use_s)

            try:
                label = int(emo_s)
            except ValueError:
                bad_labels.append({"row": i, "value": emo_s[:60]})
                continue
            if label < 0 or label > 6:
                bad_labels.append({"row": i, "value": emo_s[:60]})
                continue
            labels_seen.add(label)

            tokens = pix_s.split()
            n_tok = len(tokens)
            pixel_len_hist[n_tok] = pixel_len_hist.get(n_tok, 0) + 1
            if n_tok != IMG_PIXELS:
                bad_pixels_len.append({"row": i, "tokens": n_tok})
                continue
            try:
                for tok in tokens:
                    int(tok)
            except ValueError:
                non_integer_pixels += 1
                continue

            h = hashlib.sha1(pix_s.encode("utf-8")).hexdigest()
            if h in dup_hashes:
                duplicate_rows += 1
            else:
                dup_hashes[h] = i

            usage_counts[use_s] = usage_counts.get(use_s, 0) + 1
            class_counts[label] = class_counts.get(label, 0) + 1
            key = (use_s, label)
            usage_class_counts[key] = usage_class_counts.get(key, 0) + 1

    return {
        "header": header,
        "total_rows": total_rows,
        "usage_counts": usage_counts,
        "usages_seen": sorted(usages_seen),
        "labels_seen": sorted(labels_seen),
        "class_counts": class_counts,
        "usage_class_counts": usage_class_counts,
        "pixel_len_histogram": {str(k): v
                                for k, v in sorted(pixel_len_hist.items())},
        "bad_pixels_len": bad_pixels_len,
        "bad_labels": bad_labels,
        "bad_usage": bad_usage,
        "empty_fields": empty_fields,
        "non_integer_pixels": non_integer_pixels,
        "duplicate_rows": duplicate_rows,
        "unique_pixel_rows": len(dup_hashes),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None,
                    help="optional path to write the JSON report")
    args = ap.parse_args(argv)

    out = {"csv_path": os.path.abspath(args.csv),
           "csv_size_bytes": os.path.getsize(args.csv)}

    s = scan(args.csv)
    header = s["header"]
    out["header"] = header
    out["has_emotion"] = "emotion" in header
    out["has_pixels"] = "pixels" in header
    out["has_usage"] = "Usage" in header
    out["expected_columns_present"] = bool(
        out["has_emotion"] and out["has_pixels"] and out["has_usage"])

    out["total_rows"] = s["total_rows"]
    out["valid_rows"] = sum(s["usage_counts"].values())
    out["usage_counts"] = s["usage_counts"]
    out["usages_seen"] = s["usages_seen"]
    out["expected_usages_present"] = s["usages_seen"] == sorted(KNOWN_USAGES)
    out["labels_seen"] = s["labels_seen"]
    out["labels_valid_0_to_6"] = s["labels_seen"] == list(range(7))
    out["class_counts_by_index"] = {str(k): v
                                     for k, v in sorted(s["class_counts"].items())}
    out["class_counts_by_name"] = {EMOTIONS[k]: v
                                   for k, v in sorted(s["class_counts"].items())}
    out["usage_class_counts"] = {
        f"{u}|{EMOTIONS[c]}": n
        for (u, c), n in sorted(s["usage_class_counts"].items())}
    out["pixel_len_histogram"] = s["pixel_len_histogram"]
    out["rows_with_wrong_pixel_count"] = len(s["bad_pixels_len"])
    out["examples_wrong_pixel_count"] = s["bad_pixels_len"][:5]
    out["rows_with_bad_label"] = len(s["bad_labels"])
    out["examples_bad_label"] = s["bad_labels"][:5]
    out["rows_with_bad_usage"] = len(s["bad_usage"])
    out["examples_bad_usage"] = s["bad_usage"][:5]
    out["rows_with_non_integer_pixels"] = s["non_integer_pixels"]
    out["rows_with_missing_fields"] = s["empty_fields"]
    out["duplicate_rows"] = s["duplicate_rows"]
    out["unique_pixel_rows"] = s["unique_pixel_rows"]
    out["image_size_expected"] = f"{IMG_SIZE}x{IMG_SIZE}"
    out["pixels_per_image_expected"] = IMG_PIXELS

    # Cross-check against the project's own loader (read-only).
    records, report = load_fer_csv(args.csv)
    out["project_loader"] = {
        "kept": report.kept,
        "skipped": report.skipped,
        "total_seen": report.total_seen,
        "class_counts": report.class_counts,
        "problem_count": len(report.problems),
        "problem_sample": report.problems[:5],
    }
    out["project_loader_kept_matches_stream"] = (report.kept == out["valid_rows"])

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print("report written:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
