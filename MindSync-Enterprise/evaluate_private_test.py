# -*- coding: utf-8 -*-
"""Final PrivateTest evaluation - runs ONCE after training + promotion.

Uses the project's own evaluation module (evaluation/evaluate_model.py) so the
numbers come from the same code path the project ships. Writes a JSON report
and a confusion-matrix PNG.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(ROOT, "docs", "fer2013_private_test_metrics.json")
OUT_PNG = os.path.join(ROOT, "docs", "fer2013_private_test_confusion.png")


def main() -> int:
    from ai_services.video_facial_emotion_behavior.evaluation.evaluate_model import (
        run_evaluation,
    )
    from ai_services.video_facial_emotion_behavior.emotion_recognition.labels import (
        EMOTIONS,
    )

    metrics = run_evaluation(
        csv_path=os.path.join(ROOT, "data", "fer2013", "fer2013.csv"),
        ckpt_path=os.path.join(ROOT, "models", "fer2013_cnn_best.pt"),
        out_json=OUT_JSON,
        cm_png=OUT_PNG,
    )
    print("LABEL ORDER:", list(EMOTIONS))
    print("ACCURACY      : %.4f" % metrics["accuracy"])
    print("MACRO PREC    : %.4f" % metrics["macro_precision"])
    print("MACRO RECALL  : %.4f" % metrics["macro_recall"])
    print("MACRO F1      : %.4f" % metrics["macro_f1"])
    print("N TEST        : %d" % metrics["n_test"])
    print("MS/FACE (CPU) : %.3f" % metrics["ms_per_face_cpu"])
    print("PER-CLASS:")
    for name in EMOTIONS:
        pc = metrics["per_class"][name]
        print("  %-9s P=%.4f R=%.4f F1=%.4f"
              % (name, pc["precision"], pc["recall"], pc["f1"]))
    print("CONFUSION MATRIX (rows=true, cols=pred, order=%s):" % list(EMOTIONS))
    for i, row in enumerate(metrics["confusion_matrix"]):
        print("  %-9s %s" % (EMOTIONS[i], row))
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
