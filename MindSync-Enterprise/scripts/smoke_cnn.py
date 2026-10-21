"""Step-4 smoke test (synthetic data only, max 2 epochs, CPU).

Builds a tiny FER2013-style CSV in temp dir (no download), trains FerCNN
for <=2 epochs, checks: 7-logit output, 7-prob inference in EMOTIONS order,
BGR crop compatibility, checkpoint save/load, ModelNotAvailableError when
weights missing. Prints measured results only — NOT accuracy evidence.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    import numpy as np

    from ai_services.video_facial_emotion_behavior.emotion_recognition import \
        predict as P
    from ai_services.video_facial_emotion_behavior.emotion_recognition \
        .classifier import ModelNotAvailableError, validate_probs
    from ai_services.video_facial_emotion_behavior.emotion_recognition \
        .cnn_model import build_model, preprocess_crop
    from ai_services.video_facial_emotion_behavior.emotion_recognition \
        .fer2013 import IMG_PIXELS
    from ai_services.video_facial_emotion_behavior.emotion_recognition \
        .labels import EMOTIONS
    from ai_services.video_facial_emotion_behavior.emotion_recognition \
        .train_fer import run_training

    print("labels order:", list(EMOTIONS))
    assert list(EMOTIONS) == ["Angry", "Disgust", "Fear", "Happy", "Sad",
                              "Surprise", "Neutral"], "labels.py order changed!"

    # 1) Synthetic CSV: 7 classes x 12 Training + 7 x 3 PrivateTest.
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "synth_fer.csv")
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("emotion,pixels,Usage\n")
            for label in range(7):
                for _ in range(12):
                    px = rng.randint(0, 256, IMG_PIXELS)
                    fh.write(f"{label},{' '.join(map(str, px))},Training\n")
            for label in range(7):
                for _ in range(3):
                    px = rng.randint(0, 256, IMG_PIXELS)
                    fh.write(f"{label},{' '.join(map(str, px))},PrivateTest\n")
        out = os.path.join(td, "smoke_best.pt")

        # 2) Missing weights must raise (expected behaviour, not failure).
        try:
            P.load_model(os.path.join(td, "missing.pt"))
            print("ERROR: load_model should have raised")
            return 1
        except ModelNotAvailableError as e:
            print("missing-weights raises ModelNotAvailableError: OK -",
                  str(e)[:80])

        # 3) Train tiny: smoke=True forces CPU + epochs<=2 + subset.
        summary = run_training(csv_path=csv_path, out_path=out,
                               epochs=50, batch_size=16, smoke=True)
        print("smoke summary:",
              json.dumps({k: summary[k] for k in
                          ("best_val_loss", "epochs_run", "train_n", "val_n",
                           "out")}, indent=2))
        assert summary["epochs_run"] <= 2, "smoke exceeded 2 epochs!"
        assert os.path.exists(out), "checkpoint not saved"

        # 4) Forward shape: (N,7) logits.
        import torch

        m = build_model()
        m.eval()
        with torch.no_grad():
            logits = m(torch.zeros(4, 1, 48, 48))
        print("forward shape:", tuple(logits.shape))
        assert tuple(logits.shape) == (4, 7)

        # 5) BGR crop -> 48x48 gray -> 7 probs, order + sum checks.
        P._model = None  # reset global before load
        P.load_model(out)
        bgr = (rng.randint(0, 256, (90, 70, 3))).astype(np.uint8)
        t = preprocess_crop(bgr)
        print("preprocess_crop:", tuple(t.shape), float(t.min()),
              float(t.max()))
        assert tuple(t.shape) == (1, 48, 48)
        probs = P.predict_probs(bgr)
        ok, msg = validate_probs(probs)
        print("probs:", [round(float(v), 4) for v in probs], msg)
        assert ok, msg
        assert len(probs) == 7
    print("SMOKE OK: synthetic only, no download, no full training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
