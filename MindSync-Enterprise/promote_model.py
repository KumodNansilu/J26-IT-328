# -*- coding: utf-8 -*-
"""Validate the candidate artifacts, then promote them to models/.

Promotion is safe: the candidate checkpoint + metadata are validated against
the predict.py contract FIRST; only then are they copied over the canonical
models/fer2013_cnn_best.pt + models/fer2013_cnn_meta.json. The previous pair is
backed up first so a bad candidate can never destroy a known-good artifact.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(ROOT, "models", "candidate_full")
CAND_CKPT = os.path.join(CAND, "fer2013_cnn_best.pt")
CAND_META = os.path.join(CAND, "fer2013_cnn_meta.json")
FINAL_CKPT = os.path.join(ROOT, "models", "fer2013_cnn_best.pt")
FINAL_META = os.path.join(ROOT, "models", "fer2013_cnn_meta.json")


def main() -> int:
    from ai_services.video_facial_emotion_behavior.emotion_recognition import (
        predict as P,
    )
    from ai_services.video_facial_emotion_behavior.emotion_recognition.labels import (
        EMOTIONS,
        NUM_EMOTIONS,
    )

    if not os.path.exists(CAND_CKPT) or not os.path.exists(CAND_META):
        raise SystemExit("candidate artifacts missing - training did not finish")
    with open(CAND_META, encoding="utf-8") as fh:
        meta = json.load(fh)
    print("CANDIDATE META:", json.dumps(meta, indent=2))
    if meta.get("smoke"):
        raise SystemExit("refusing to promote a smoke artifact")
    assert meta["label_order"] == list(EMOTIONS)
    assert meta["input"] == [1, 48, 48]
    assert meta["epochs_run"] >= 1

    # Validate the CANDIDATE pair together (predict.py resolves the metadata
    # next to the checkpoint, so they must be validated as a pair).
    P._model = None
    info = P.load_model(CAND_CKPT)
    print("CANDIDATE LOAD OK:", info)

    import numpy as np
    bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
    probs = P.predict_probs(bgr)
    assert len(probs) == NUM_EMOTIONS
    assert abs(sum(probs) - 1.0) < 1e-5
    assert all(np.isfinite(probs))
    print("CANDIDATE INFERENCE OK: probs=%s" % [round(p, 4) for p in probs])

    # Promote: back up the previous canonical pair, then copy.
    if os.path.exists(FINAL_CKPT):
        shutil.copy2(FINAL_CKPT, FINAL_CKPT + ".bak")
        print("backed up previous checkpoint")
    if os.path.exists(FINAL_META):
        shutil.copy2(FINAL_META, FINAL_META + ".bak")
        print("backed up previous metadata")
    shutil.copy2(CAND_CKPT, FINAL_CKPT)
    shutil.copy2(CAND_META, FINAL_META)
    print("PROMOTED to", FINAL_CKPT)

    # Re-validate the promoted pair from its canonical location.
    P._model = None
    info2 = P.load_model(FINAL_CKPT)
    print("PROMOTED LOAD OK:", info2)
    probs2 = P.predict_probs(bgr)
    assert len(probs2) == NUM_EMOTIONS
    print("PROMOTED INFERENCE OK")
    for p in (FINAL_CKPT, FINAL_META):
        print("%s  %d bytes" % (p, os.path.getsize(p)))
    print("PROMOTE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
