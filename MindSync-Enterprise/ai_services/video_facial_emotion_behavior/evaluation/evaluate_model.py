"""Final evaluation on locked PrivateTest. Test set touched ONCE here.

Usage (after full training approved + done):
  python -m ...evaluate_model --csv data/fer2013.csv --ckpt models/...pt
Reports accuracy, macro/per-class P/R/F1, confusion matrix PNG, ms/face.
Never invent numbers — prints measured values only.
"""
from __future__ import annotations

import argparse
import json
import time


def run_evaluation(*, csv_path: str, ckpt_path: str, out_json: str = "",
                   cm_png: str = "") -> dict:
    import numpy as np

    from ..emotion_recognition.fer2013 import load_fer_csv, lock_test_set
    from ..emotion_recognition.labels import EMOTIONS
    from ..emotion_recognition.predict import load_model, predict_probs
    from .metrics import compute_metrics, save_confusion_matrix_png

    records, _ = load_fer_csv(csv_path)
    test = lock_test_set(records, allow_test=True)
    if not test:
        raise ValueError("No PrivateTest rows found — check the CSV Usage column.")
    load_model(ckpt_path)
    y_true, y_pred = [], []
    t0 = time.perf_counter()
    for r in test:
        import cv2

        bgr = cv2.cvtColor(r.image, cv2.COLOR_GRAY2BGR)
        probs = predict_probs(bgr)
        y_true.append(r.label)
        y_pred.append(int(np.argmax(probs)))
    secs = time.perf_counter() - t0
    metrics = compute_metrics(y_true, y_pred, labels=list(EMOTIONS))
    metrics["n_test"] = len(test)
    metrics["ms_per_face_cpu"] = secs / max(len(test), 1) * 1000.0
    if out_json:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    if cm_png:
        save_confusion_matrix_png(metrics["confusion_matrix"],
                                  list(EMOTIONS), cm_png)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--cm", default="")
    args = ap.parse_args(argv)
    print(json.dumps(run_evaluation(csv_path=args.csv, ckpt_path=args.ckpt,
                                    out_json=args.out, cm_png=args.cm),
                     indent=2))


if __name__ == "__main__":
    main()

